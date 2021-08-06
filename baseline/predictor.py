import os
import json
import multiprocessing
from itertools import chain, product
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

import common
from common import Trajectory, VideoRelation
from video_object_detection.object_tracklet_proposal import get_object_tracklets
from .feature import extract_object_feature, extract_relation_feature
from .model import IndependentClassifier, CascadeClassifier, IterativeClassifier


class TestDataset(Dataset):
    def __init__(self, raw_dataset, split, **param):
        self.raw_dataset = raw_dataset
        self.split = split
        self.temporal_propagate_threshold = param['inference_temporal_propagate_threshold']
        self.n_workers = param['inference_n_workers']
        self.video_segments = self._get_testing_segments()

    def __len__(self):
        return len(self.video_segments)

    def __getitem__(self, i):
        index = self.video_segments[i]
        vid, fstart, fend = index
        anno = self.raw_dataset.get_anno(vid)
        pairs, sub_feats, obj_feats, pred_pos_feats, pred_vis_feats, iou, trackid = extract_relation_feature(
                self.raw_dataset.name, vid, fstart, fend, anno)
        sub_feats = sub_feats.astype(np.float32)
        obj_feats = obj_feats.astype(np.float32)
        pred_pos_feats = pred_pos_feats.astype(np.float32)
        pred_vis_feats = pred_vis_feats.astype(np.float32)
        tracklets = self._get_object_tracklets(self.raw_dataset.name, vid, fstart, fend, anno)

        if self.temporal_propagate_threshold == 1:
            return index, pairs, sub_feats, obj_feats, pred_pos_feats, pred_vis_feats, iou, tracklets, None

        asso_graph = None
        if i > 0:
            last_index = self.video_segments[i-1]
            last_vid, last_fstart, last_fend = last_index
            if _is_precede_video_segment(last_index, index):
                last_tracklets = self._get_object_tracklets(self.raw_dataset.name, last_vid, last_fstart, last_fend,
                        self.raw_dataset.get_anno(last_vid))
                last_pairs = np.asarray([[ti, tj] for ti, tj in product(range(len(last_tracklets)), range(len(last_tracklets)))])
                asso_graph = []
                for ti, trac in enumerate(tracklets):
                    edges = []
                    for tj, last_trac in enumerate(last_tracklets):
                        overlap = trac.cubic_intersection(last_trac, temporal_tolerance=0)
                        if overlap > self.temporal_propagate_threshold:
                            edges.append((tj, overlap))
                    asso_graph.append(edges)

        if asso_graph is None:
            trans_mat = None
        else:
            trans_mat = np.zeros((3, pairs.shape[0], last_pairs.shape[0]), dtype=np.float32)
            last_pair2id = dict(((ti, tj), i) for i, (ti, tj) in enumerate(last_pairs))
            for pair_id, (sub_id, obj_id) in enumerate(pairs):
                for last_sub_id, sub_ov in asso_graph[sub_id]:
                    for last_obj_id, obj_ov in asso_graph[obj_id]:
                        last_pair_id = last_pair2id[(last_sub_id, last_obj_id)]
                        trans_mat[0, pair_id, last_pair_id] = sub_ov # subject
                        trans_mat[1, pair_id, last_pair_id] = obj_ov # object
                        trans_mat[2, pair_id, last_pair_id] = min(sub_ov, obj_ov) # predicate
            e_x = np.exp(trans_mat-np.max(trans_mat, axis=-1, keepdims=True))
            e_x = e_x*(trans_mat>0).astype(np.float32)
            trans_mat = e_x/np.clip(np.sum(e_x, axis=-1, keepdims=True), 1e-8, None)

        return index, pairs, sub_feats, obj_feats, pred_pos_feats, pred_vis_feats, iou, tracklets, trans_mat

    def _get_object_tracklets(self, dname, vid, fstart, fend, anno):
        tracklets = []
        _tracklets, _, _ = extract_object_feature(dname, vid, fstart, fend, anno)
        for _trac in _tracklets:
            pstart = fstart + int(_trac['fstart'])
            pend = pstart + len(_trac['bboxes'])
            trac = Trajectory(pstart, pend, _trac['bboxes'], score=_trac['score'])
            tracklets.append(trac)
        return tracklets

    def _get_testing_segments(self):
        print('[info] preparing video segments from {} set for testing'.format(self.split))
        video_segments = dict()
        video_indices = self.raw_dataset.get_index(split=self.split)

        if self.n_workers > 0:
            with tqdm(total=len(video_indices)) as pbar:
                pool = multiprocessing.Pool(processes=self.n_workers)
                for vid in video_indices:
                    anno = self.raw_dataset.get_anno(vid)
                    video_segments[vid] = pool.apply_async(_get_testing_segments_for_video,
                            args=(self.raw_dataset.name, vid, anno),
                            callback=lambda _: pbar.update())
                pool.close()
                pool.join()
            for vid in video_segments.keys():
                res = video_segments[vid].get()
                video_segments[vid] = res
        else:
            for vid in tqdm(video_indices):
                anno = self.raw_dataset.get_anno(vid)
                res = _get_testing_segments_for_video(self.raw_dataset.name, vid, anno)
                video_segments[vid] = res
            
        return list(chain.from_iterable(video_segments.values()))


def _is_precede_video_segment(last_index, index):
    return last_index[0] == index[0] and last_index[2] >= index[1]


def _get_testing_segments_for_video(dname, vid, anno):
    video_segments = []
    segs = common.segment_video(0, anno['frame_count'])
    for fstart, fend in segs:
        tracklets, _, _ = extract_object_feature(dname, vid, fstart, fend, anno)
        # if multiple objects detected and the relation features extracted
        if len(tracklets) > 1:
            video_segments.append((vid, fstart, fend))
    return video_segments


@torch.no_grad()
def predict(raw_dataset, split, use_cuda=False, output_json=True, **param):
    test_dataset = TestDataset(raw_dataset, split, **param)
    data_generator = DataLoader(test_dataset, batch_size=1, num_workers=param['inference_n_workers'], collate_fn=lambda bs: bs[0])

    model_path = os.path.join(common.get_model_path(param['exp_id'], param['dataset']), 'weights',
            '{}{}'.format(param['model'].get('dump_file_prefix', ''), param['model']['dump_file']))
    print('[info] loading model from file: {}'.format(model_path))
    if param['model']['name'] == 'independent_classifier':
        model = IndependentClassifier(**param)
    elif param['model']['name'] == 'cascade_classifier':
        model = CascadeClassifier(**param)
    elif param['model']['name'] == 'iterative_classifier':
        model = IterativeClassifier(**param)
    else:
        raise ValueError(param['model']['name'])
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, location: storage))
    model.infer_zero_shot_preference(strategy=param['model'].get('zero_shot_preference', 'none'))
    if use_cuda:
        model.cuda()

    print('[info] predicting visual relation segments')
    model.eval()
    relation_segments = dict()
    for data in tqdm(data_generator):
        index, pairs, sub_feats, obj_feats, pred_pos_feats, pred_vis_feats, iou, tracklets, trans_mat = data
        
        pairs = torch.from_numpy(pairs)
        sub_feats = torch.from_numpy(sub_feats)
        obj_feats = torch.from_numpy(obj_feats)
        pred_pos_feats = torch.from_numpy(pred_pos_feats)
        pred_vis_feats = torch.from_numpy(pred_vis_feats)
        trans_mat = None if trans_mat is None else torch.from_numpy(trans_mat)
        if use_cuda:
            pairs = pairs.cuda()
            sub_feats = sub_feats.cuda()
            obj_feats = obj_feats.cuda()
            pred_pos_feats = pred_pos_feats.cuda()
            pred_vis_feats = pred_vis_feats.cuda()
            trans_mat = None if trans_mat is None else trans_mat.cuda()
        
        model_predictions = model.predict(pairs, sub_feats, obj_feats, pred_pos_feats, pred_vis_feats, trans_mat=trans_mat,
                inference_steps=param['inference_steps'],
                inference_problistic=param['inference_problistic'],
                inference_object_conf_thres=param['inference_object_conf_threshold'],
                inference_predicate_conf_thres=param['inference_predicate_conf_threshold'])

        # supression
        model_predictions = sorted(model_predictions, key=lambda r: r['score'], reverse=True)[:param['inference_topk']]
        if param['inference_nms'] < 1:
            model_predictions = relation_nms(model_predictions, iou, param['inference_nms'])

        predictions = []
        for r in model_predictions:
            sub = raw_dataset.get_object_name(r['triplet'][0])
            pred = raw_dataset.get_predicate_name(r['triplet'][1])
            obj = raw_dataset.get_object_name(r['triplet'][2])
            predictions.append(VideoRelation(sub, pred, obj, tracklets[r['sub_id']], tracklets[r['obj_id']], r['score']))
        
        vsig = common.get_segment_signature(*index)
        if output_json:
            relation_segments[vsig] = [r.serialize(allow_misalign=True) for r in predictions]
        else:
            relation_segments[vsig] = predictions

    return relation_segments


def relation_nms(relations, iou, suppress_threshold=0.9, max_n_return=None):
    if len(relations) == 0:
        return []
    
    order = sorted(range(len(relations)), key=lambda i: relations[i]['score'], reverse=True)
    if max_n_return is None:
        max_n_return = len(order)

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(relations[i])
        if len(keep) >= max_n_return:
            break
        triplet = relations[i]['triplet']
        sub_id, obj_id = relations[i]['sub_id'], relations[i]['obj_id']
        new_order = []
        for j in order:
            supress = False
            if triplet == relations[j]['triplet']:
                sub_id_j, obj_id_j = relations[j]['sub_id'], relations[j]['obj_id']
                supress = iou[sub_id_j, sub_id]>suppress_threshold and iou[obj_id_j, obj_id]>suppress_threshold
            if not supress:
                new_order.append(j)
        order = new_order

    return keep
