import os
import pickle
import time
import random
import multiprocessing
from itertools import product, chain
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, get_worker_info

import common
from .feature import extract_object_feature, extract_relation_feature
from .model import IndependentClassifier, CascadeClassifier, IterativeClassifier
from .misc import FocalLoss, binary_focal_loss


class TrainDataset(Dataset):
    def __init__(self, raw_dataset, split, rng, negative_positive_ratio=1.0, **param):
        super().__init__()
        self.raw_dataset = raw_dataset
        self.split = split
        self.rng = rng
        self.max_cand_per_sample = param['training_max_cand_per_sample']
        self.negative_positive_ratio = negative_positive_ratio
        self.viou_threshold = param['training_poi_threshold']
        self.negative_viou_threshold = param['training_negative_poi_threshold']
        self.time_overlap_threshold = param['training_time_overlap_threshold']
        self.ignore_segments_wo_gt = param['training_ignore_segments_wo_gt']
        self.include_gt_tracklet = param['training_use_gt_tracklet']
        self.n_workers = param['training_n_workers']

        self.seg_cache_path = os.path.join(common.get_model_path(param['exp_id'], param['dataset']), 'training_segment_cache.pkl')
        self.sample_cache_path = os.path.join(common.get_model_path(param['exp_id'], param['dataset']), 'training_sample_cache.pkl')

        if param['use_cached_training_sample'] and os.path.exists(self.sample_cache_path):
            print('[info] loading training sample cache from {}'.format(self.sample_cache_path))
            with open(self.sample_cache_path, 'rb') as fin:
                cache = pickle.load(fin)
                self.samples = cache['samples']
                self.num_ignored_segments = cache['num_ignored_segments']
        else:
            if param['use_cached_training_sample'] and os.path.exists(self.seg_cache_path):
                print('[info] loading training segment cache from {}'.format(self.seg_cache_path))
                with open(self.seg_cache_path, 'rb') as fin:
                    cache = pickle.load(fin)
                    self.samples = cache['samples']
                    self.num_ignored_segments = cache['num_ignored_segments']
            else:
                self.samples, self.num_ignored_segments = self._get_training_segments()
                with open(self.seg_cache_path, 'wb') as fout:
                    pickle.dump({
                        'samples': self.samples,
                        'num_ignored_segments': self.num_ignored_segments
                    }, fout)
            self.samples = self._preprocessing_training_samples(self.samples)
            with open(self.sample_cache_path, 'wb') as fout:
                pickle.dump({
                    'samples': self.samples,
                    'num_ignored_segments': self.num_ignored_segments
                }, fout)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        num_to_sample = dict()
        sample['all_candidates'] = sample['pos_candidates']+sample['neg_candidates']
        num_to_sample['all_candidates'] = min(self.max_cand_per_sample, len(sample['all_candidates']))

        s_batch = []
        p_vec_batch = []
        o_batch = []
        sampled_pair_ids = []
        for cand_type in ['all_candidates']:
            if len(sample[cand_type])>0 and num_to_sample[cand_type]>0:
                indices = self.rng.choice(len(sample[cand_type]), num_to_sample[cand_type], replace=False)
            else:
                indices = []
            for i in indices:
                cand = sample[cand_type][i]
                sampled_pair_ids.append(cand['pair_id'])
                s_batch.append(cand['sub_class_id'])
                p_vec = np.zeros((self.raw_dataset.get_predicate_num(),), dtype=np.float32)
                for p in cand['pred_class_ids']:
                    p_vec[p] = 1
                p_vec_batch.append(p_vec)
                o_batch.append(cand['obj_class_id'])

        _, sf_batch, of_batch, ppf_batch, pvf_batch, _, _ = extract_relation_feature(
                self.raw_dataset.name, *sample['index'],
                self.raw_dataset.get_anno(sample['index'][0]),
                sampled_pair_ids=sampled_pair_ids, include_gt=self.include_gt_tracklet)

        return sf_batch, of_batch, ppf_batch, pvf_batch,\
                np.asarray(s_batch), np.stack(p_vec_batch), np.asarray(o_batch)

    def _get_training_segments(self):
        print('[info] preparing video segments from {} set for training'.format(self.split))
        video_segments = dict()
        num_ignored = 0
        video_indices = self.raw_dataset.get_index(split=self.split)
        obj_cls2id = self.raw_dataset.so2soid
        pred_cls2id = self.raw_dataset.pred2pid

        if self.n_workers > 0:
            with tqdm(total=len(video_indices)) as pbar:
                pool = multiprocessing.Pool(processes=self.n_workers)
                for vid in video_indices:
                    anno = self.raw_dataset.get_anno(vid)
                    rel_insts = self.raw_dataset.get_relation_insts(vid, no_traj=True)
                    video_segments[vid] = pool.apply_async(_get_training_segments_for_video,
                            args=(self.raw_dataset.name, vid, obj_cls2id, pred_cls2id, anno, rel_insts,
                                    self.time_overlap_threshold, self.ignore_segments_wo_gt, self.include_gt_tracklet),
                            callback=lambda _: pbar.update())
                pool.close()
                pool.join()
            for vid in video_segments.keys():
                res = video_segments[vid].get()
                video_segments[vid] = res[0]
                num_ignored += res[1]
        else:
            for vid in tqdm(video_indices):
                anno = self.raw_dataset.get_anno(vid)
                rel_insts = self.raw_dataset.get_relation_insts(vid, no_traj=True)
                res = _get_training_segments_for_video(
                        self.raw_dataset.name, vid, obj_cls2id, pred_cls2id, anno, rel_insts,
                        time_overlap_threshold=self.time_overlap_threshold,
                        ignore_segments_wo_gt=self.ignore_segments_wo_gt,
                        include_gt_tracklet=self.include_gt_tracklet)
                video_segments[vid] = res[0]
                num_ignored += res[1]

        samples = list(chain.from_iterable(video_segments.values()))
        if self.ignore_segments_wo_gt:
            print('[info] ignore {} video segments without any ground truth relations'.format(num_ignored))
        return samples, num_ignored

    def _preprocessing_training_samples(self, samples):
        print('[info] preprocessing training samples')
        num_pos_candicates = []
        num_neg_candicates = []
        num_unique_triplets = []
        keep = []

        if self.n_workers > 0:
            with tqdm(total=len(samples)) as pbar:
                pool = multiprocessing.Pool(processes=self.n_workers)
                results = []
                for sample in samples:
                    if 'pos_candidates' not in sample and 'neg_candidates' not in sample:
                        anno = self.raw_dataset.get_anno(sample['index'][0])
                        results.append(pool.apply_async(_preprocessing_training_sample,
                                args=(self.raw_dataset.name, anno, sample,
                                    self.viou_threshold, self.negative_viou_threshold,
                                    self.include_gt_tracklet),
                                callback=lambda _: pbar.update()))
                    else:
                        results.append(sample)
                pool.close()
                pool.join()
            for res in results:
                if isinstance(res, multiprocessing.pool.AsyncResult):
                    res = res.get()
                if res is None:
                    continue
                num_pos_candicates.append(len(res.get('pos_candidates', [])))
                num_neg_candicates.append(len(res.get('neg_candidates', [])))
                num_unique_triplets.append(len(res['unique_triplets']))
                keep.append(res)
        else:
            for sample in tqdm(samples):
                if 'pos_candidates' not in sample and 'neg_candidates' not in sample:
                    anno = self.raw_dataset.get_anno(sample['index'][0])
                    res = _preprocessing_training_sample(self.raw_dataset.name, anno, sample,
                            viou_threshold=self.viou_threshold, negative_viou_threshold=self.negative_viou_threshold,
                            include_gt_tracklet=self.include_gt_tracklet)
                    if res is None:
                        continue
                num_pos_candicates.append(len(res.get('pos_candidates', [])))
                num_neg_candicates.append(len(res.get('neg_candidates', [])))
                num_unique_triplets.append(len(res['unique_triplets']))
                keep.append(res)

        print('[info] get {} positive and {} negative candidate pairs for training'.format(
                np.sum(num_pos_candicates), np.sum(num_neg_candicates)))
        print('------ mean number of positive and negative candidates per sample: {:.1f} and {:.1f}'.format(
                np.mean(num_pos_candicates), np.mean(num_neg_candicates)))
        print('------ median number of positive and negative candidates per sample: {} and {}'.format(
                np.median(num_pos_candicates), np.median(num_neg_candicates)))
        print('------ mean (std) and maximum number of unique_triplets per sample: {:.1f} ({:.1f}) and {}'.format(
                np.mean(num_unique_triplets), np.std(num_unique_triplets), np.max(num_unique_triplets)))
        return keep


def _get_training_segments_for_video(dname, vid, obj_cls2id, pred_cls2id, anno, rel_insts,
        time_overlap_threshold=1, ignore_segments_wo_gt=True, include_gt_tracklet=True):
    video_segments = []
    num_ignored = 0
    segs = common.segment_video(0, anno['frame_count'])
    tid2cls = dict((x['tid'], x['category']) for x in anno['subject/objects'])
    for fstart, fend in segs:
        observed = []
        for rel_inst in rel_insts:
            max_start = max(rel_inst['duration'][0], fstart)
            min_end = min(rel_inst['duration'][1], fend)
            if min_end - max_start >= time_overlap_threshold:
                observed.append(rel_inst)
        if ignore_segments_wo_gt and len(observed) == 0:
            num_ignored += 1
            continue
        tracklets, _, _ = extract_object_feature(dname, vid, fstart, fend, anno, include_gt=include_gt_tracklet)
        num_tracklet_with_feature = 0
        for trac in tracklets:
            if trac['feature'].size > 0:
                num_tracklet_with_feature += 1
        # if multiple objects detected and the relation features extracted
        if num_tracklet_with_feature > 1:
            video_segments.append({
                'index': (vid, fstart, fend),
                'gt_relations': observed,
                'trackid_class': tid2cls,
                'object_class_id': obj_cls2id,
                'predicate_class_id': pred_cls2id
            })
    return video_segments, num_ignored


def _preprocessing_training_sample(dname, anno, sample, viou_threshold=0.9, negative_viou_threshold=0.5, include_gt_tracklet=True):
    vid, fstart, fend = sample['index']
    tracklets, track_gt_id, pairwise_iou = extract_object_feature(dname, vid, fstart, fend, anno, include_gt=include_gt_tracklet)
    num_tracklets = len(tracklets)
    pairs = np.asarray([[ti, tj] for ti, tj in product(range(num_tracklets), range(num_tracklets))])
    pair2id = dict(((trac1, trac2), i) for i, (trac1, trac2) in enumerate(pairs))
    observed = defaultdict(list)
    for rel_inst in sample['gt_relations']:
        observed[(rel_inst['subject_tid'], rel_inst['object_tid'])].append(rel_inst)
    tid2cls = sample['trackid_class']
    obj_cls2id = sample['object_class_id']
    pred_cls2id = sample['predicate_class_id']

    pos_candidates = []
    pos_pair_ids = set()
    unique_triplets = set()
    for (subject_tid, object_tid), rel_insts in observed.items():
        si = oi = None
        for i, tid in enumerate(track_gt_id):
            if tid == subject_tid:
                si = i
            elif tid == object_tid:
                oi = i
        if si is None or oi is None:
            continue
        
        s_iou = pairwise_iou[:, si]
        o_iou = pairwise_iou[:, oi]
        s_inds = np.where(s_iou>viou_threshold)[0]
        o_inds = np.where(o_iou>viou_threshold)[0]
        pair_ids = [pair2id[(ti, tj)] for ti, tj in product(s_inds, o_inds) if ti != tj]
        pos_pair_ids.update(pair_ids)

        sub_class_id = obj_cls2id[tid2cls[subject_tid]]
        pred_class_ids = [pred_cls2id[r['triplet'][1]] for r in rel_insts]
        obj_class_id = obj_cls2id[tid2cls[object_tid]]
        for pair_id in pair_ids:
            ti, tj = pairs[pair_id]
            if tracklets[ti]['feature'].size == 0:
                assert track_gt_id[ti] > -1
                # feature not available (for gt tracklets)
                continue
            if tracklets[tj]['feature'].size == 0:
                assert track_gt_id[tj] > -1
                continue
            pos_candidates.append({
                'pair_id': pair_id,
                'sub_class_id': sub_class_id,
                'pred_class_ids': pred_class_ids,
                'obj_class_id': obj_class_id
            })
            unique_triplets.update((sub_class_id, pid, obj_class_id) for pid in pred_class_ids)

    neg_candidates = []
    num_det = len([i for i in track_gt_id if i < 0])
    neg_pair_ids = set(range(len(pairs)))-pos_pair_ids
    for pair_id in neg_pair_ids:
        ti, tj = pairs[pair_id]
        if tracklets[ti]['feature'].size == 0:
            assert track_gt_id[ti] > -1
            continue
        if tracklets[tj]['feature'].size == 0:
            assert track_gt_id[tj] > -1
            continue

        si, oi = pairs[pair_id]
        max_gt_si = np.argmax(pairwise_iou[si, num_det:])+num_det
        max_gt_oi = np.argmax(pairwise_iou[oi, num_det:])+num_det
        max_gt_siou = pairwise_iou[si, max_gt_si]
        max_gt_oiou = pairwise_iou[oi, max_gt_oi]

        if max_gt_siou > viou_threshold and max_gt_oiou > viou_threshold:
            sub_class_id = obj_cls2id[tid2cls[track_gt_id[max_gt_si]]]
            obj_class_id = obj_cls2id[tid2cls[track_gt_id[max_gt_oi]]]
            pos_candidates.append({
                'pair_id': pair_id,
                'sub_class_id': sub_class_id,
                'pred_class_ids': [],
                'obj_class_id': obj_class_id
            })
            unique_triplets.add((sub_class_id, -1, obj_class_id))

        elif max_gt_siou < negative_viou_threshold or max_gt_oiou < negative_viou_threshold:
            if max_gt_siou > viou_threshold:
                sub_name = tid2cls[track_gt_id[max_gt_si]]
                sub_class_id = obj_cls2id[sub_name]
            else:
                sub_class_id = len(obj_cls2id)
            if max_gt_oiou > viou_threshold:
                obj_name = tid2cls[track_gt_id[max_gt_oi]]
                obj_class_id = obj_cls2id[obj_name]
            else:
                obj_class_id = len(obj_cls2id)

            neg_candidates.append({
                'pair_id': pair_id,
                'sub_class_id': sub_class_id,
                'pred_class_ids': [],
                'obj_class_id': obj_class_id
            })
            unique_triplets.add((sub_class_id, -1, obj_class_id))
    
    if len(pos_candidates)>0 or len(neg_candidates):
        sample['pos_candidates'] = pos_candidates
        sample['neg_candidates'] = neg_candidates
        sample['unique_triplets'] = unique_triplets
        return sample
    else:
        return None


def _train_worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    dataset.rng = np.random.default_rng(worker_info.seed)


def _collate_fn(batch_list):
    items = []
    for item_id in range(len(batch_list[0])):
        item = [x[item_id] for x in batch_list]
        if isinstance(item[0], np.ndarray):
            item = np.concatenate(item)
            item = torch.from_numpy(item)
        items.append(item)
    return items


def _init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def compute_component_dependency(samples, raw_dataset, **param):
    object_num = param['object_num']+1
    predicate_num = param['predicate_num']
    count = np.zeros((object_num, predicate_num, object_num), dtype=np.float32)
    for vid in raw_dataset.get_index(split=param['train_split']):
        rel_insts = raw_dataset.get_relation_insts(vid, no_traj=True)
        for r in rel_insts:
            s, p, o = r['triplet']
            s = raw_dataset.get_object_id(s)
            p = raw_dataset.get_predicate_id(p)
            o = raw_dataset.get_object_id(o)
            count[s, p, o] += 1
    sp2o = np.divide(count+param['model']['smoothing'], np.sum(count, axis=2, keepdims=True)+param['model']['smoothing']*object_num, dtype=np.float32)
    sp2o = sp2o.reshape((-1, object_num))
    po2s = np.divide(count+param['model']['smoothing'], np.sum(count, axis=0, keepdims=True)+param['model']['smoothing']*object_num, dtype=np.float32)
    po2s = np.transpose(po2s, (1, 2, 0)).reshape((-1, object_num))
    so2p = np.divide(count+param['model']['smoothing'], np.sum(count, axis=1, keepdims=True)+param['model']['smoothing']*2, dtype=np.float32)
    so2p = np.transpose(so2p, (0, 2, 1)).reshape((-1, predicate_num))

    return sp2o, po2s, so2p


def train(raw_dataset, split, use_cuda=False, **param):
    param['object_num'] = raw_dataset.get_object_num()
    param['predicate_num'] = raw_dataset.get_predicate_num()

    train_dataset = TrainDataset(raw_dataset, split, np.random.default_rng(param['rng_seed']), **param)
    sf, of, ppf, pvf, _, _, _ = train_dataset[0]
    param['sub_feature_dim'] = sf.shape[1]
    param['obj_feature_dim'] = of.shape[1]
    param['pred_pos_feature_dim'] = ppf.shape[1]
    param['pred_vis_feature_dim'] = pvf.shape[1]
    print('[info] feature dimension for subject: {}, object: {}, predicate positional: {}, predicate visual: {}'.format(
            param['sub_feature_dim'], param['obj_feature_dim'], param['pred_pos_feature_dim'], param['pred_vis_feature_dim']))

    data_generator = DataLoader(train_dataset, batch_size=param['training_batch_size'], shuffle=True,
            num_workers=param['training_n_workers'], worker_init_fn=_train_worker_init_fn, collate_fn=_collate_fn)

    if param['model']['name'] == 'independent_classifier':
        model = IndependentClassifier(**param)
    elif param['model']['name'] == 'cascade_classifier':
        if param['model']['use_knowledge']:
            with open(train_dataset.seg_cache_path, 'rb') as fin:
                print('[info] computing triplet component dependency')
                _, _, so2p = compute_component_dependency(pickle.load(fin)['samples'], raw_dataset, **param)
            model = CascadeClassifier(so2p=so2p, **param)
        else:
            model = CascadeClassifier(**param)
    elif param['model']['name'] == 'iterative_classifier':
        if param['model']['use_knowledge']:
            with open(train_dataset.seg_cache_path, 'rb') as fin:
                print('[info] computing triplet component dependency')
                sp2o, po2s, so2p = compute_component_dependency(pickle.load(fin)['samples'], raw_dataset, **param)
            model = IterativeClassifier(sp2o=sp2o, po2s=po2s, so2p=so2p, **param)
        else:
            model = IterativeClassifier(**param)
    else:
        raise ValueError(param['model']['name'])
    print(model)
    model.apply(_init_weights)
    if use_cuda:
        model.cuda()
    visual_parameters = []
    preferential_parameters = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if name in ['sp2o', 'po2s', 'so2p']:
                preferential_parameters.append(p)
            else:
                visual_parameters.append(p)
    visual_optim = optim.AdamW(visual_parameters, lr=param['training_lr'], weight_decay=param['training_weight_decay'])
    if len(preferential_parameters) > 0:
        preferential_optim = optim.AdamW(preferential_parameters, lr=param['training_lr'], weight_decay=param['model']['weight_decay'])
    focal_loss = FocalLoss(gamma=param['training_focal_gamma'])

    best_loss = float('inf')
    best_ep = 0
    for ep in range(1, param['training_max_epoch']+1):
        num_iter = len(data_generator)
        s_losses = []
        p_losses = []
        o_losses = []
        total_losses = []
        print('\nEpoch {}'.format(ep))
        model.train()
        t_epoch_start = time.time()
        t_iter_start = time.time()
        for it, data in enumerate(data_generator):
            sf, of, ppf, pvf, s, p_vec, o = data
            if use_cuda:
                sf, of, ppf, pvf, s, p_vec, o = sf.cuda(), of.cuda(), ppf.cuda(), pvf.cuda(), s.cuda(), p_vec.cuda(), o.cuda()

            t_model_start = time.time()
            visual_optim.zero_grad()
            if len(preferential_parameters) > 0:
                preferential_optim.zero_grad()
            s_score, o_score, p_score = model(sf, of, ppf, pvf, gt_s=s, gt_o=o, gt_p_vec=p_vec)
            s_loss = focal_loss(s_score, s)
            o_loss = focal_loss(o_score, o)
            p_loss = binary_focal_loss(p_score, p_vec, param['training_focal_gamma'])
            total_loss = s_loss+o_loss+p_loss
            total_loss.backward()
            visual_optim.step()
            if len(preferential_parameters) > 0:
                preferential_optim.step()

            s_losses.append(s_loss.data.item())
            o_losses.append(o_loss.data.item())
            p_losses.append(p_loss.data.item())
            total_losses.append(total_loss.data.item())
            t_model_end = time.time()

            if it % param['training_display_freq'] == 0:
                print('iter: [{}/{}], s_loss: {:.4f}, o_loss: {:.4f}, p_loss: {:.4f}, total: {:.4f}, data time: {:.4f}s, total time: {:.4f}s'.format(
                        it, num_iter,
                        s_loss.data.item(), o_loss.data.item(), p_loss.data.item(), total_loss.data.item(),
                        t_model_start-t_iter_start, t_model_end-t_iter_start), end='\r')
            t_iter_start = time.time()

        print('-'*120)
        print('Training summary:'.format(ep))
        print('[info] s_loss: {:.4f}, o_loss: {:.4f}, p_loss: {:.4f}, total: {:.4f}, time: {:.4f}s'.format(
                np.mean(s_losses), np.mean(o_losses), np.mean(p_losses),
                np.mean(total_losses), time.time()-t_epoch_start))
        
        if ep % param['training_save_freq'] == 0:
            dump_path = os.path.join(common.get_model_path(param['exp_id'], param['dataset']), 'weights',
                    '{}weights_ep{}.t7'.format(param['model'].get('dump_file_prefix', ''), ep))
            print('[info] saving weights to {}'.format(dump_path))
            torch.save(model.state_dict(), dump_path)
        
        if np.mean(total_losses) < best_loss:
            best_loss = np.mean(total_losses)
            best_ep = ep
            dump_path = os.path.join(common.get_model_path(param['exp_id'], param['dataset']), 'weights',
                    '{}weights_best.t7'.format(param['model'].get('dump_file_prefix', '')))
            print('[info] saving weights to {}'.format(dump_path))
            torch.save(model.state_dict(), dump_path)

    print('[info] best training loss {:.4f} after {} training epoch'.format(best_loss, best_ep))
    return param
