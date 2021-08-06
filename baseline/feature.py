import os
from itertools import product

import numpy as np
import h5py

import common
from common import cubic_iou
from video_object_detection.object_tracklet_proposal import get_object_tracklets


_dummy_tiny_bbox = 0.00, 0.00, 0.001, 0.001


def _convert_bbox(bbox):
    x = (bbox[0]+bbox[2])/2
    y = (bbox[1]+bbox[3])/2
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    return x, y, w, h


def _compute_relative_positional_feature(subj_bbox, t_subj_bbox, obj_bbox, t_obj_bbox):
    subj_x, subj_y, subj_w, subj_h = _convert_bbox(subj_bbox)
    obj_x, obj_y, obj_w, obj_h = _convert_bbox(obj_bbox)
    rx = (subj_x-obj_x)/obj_w
    ry = (subj_y-obj_y)/obj_h
    log_subj_w, log_subj_h = np.log(subj_w), np.log(subj_h)
    log_obj_w, log_obj_h = np.log(obj_w), np.log(obj_h)
    rw = log_subj_w-log_obj_w
    rh = log_subj_h-log_obj_h
    ra = log_subj_w+log_subj_h-log_obj_w-log_obj_h
    return np.asarray([rx, ry, rw, rh, ra, (t_subj_bbox-t_obj_bbox)/common.segment_length], dtype=np.float32)


def extract_object_feature(dname, vid, fstart, fend, anno, include_gt=False, verbose=False):
    vsig = common.get_segment_signature(vid, fstart, fend)
    path = common.get_feature_path(dname, 'object', anno.get('video_path', vid).replace('.mp4', ''))
    path = os.path.join(path, '{}-{}.h5'.format(vsig, 'object'))
    if os.path.exists(path):
        if verbose:
            print('loading object feature for video segment {}...'.format(vsig))
        try:
            with h5py.File(path, 'r') as fin:
                # N object trajectory proposals, whose trackids are all -1
                # and M groundtruth object trajectories, whose trackids are provided by dataset
                track_gt_id = fin['track_gt_id'][:]
                # vIoU (traj_iou) for each pair (in same order)
                pairwise_iou = fin['pairwise_iou'][:]
                # positional and visual feature for each tracklet
                tracklets = []
                for ti in range(len(track_gt_id)):
                    tracklets.append({
                        'fstart': fin['/{}/fstart'.format(ti)][()],
                        'score': fin['/{}/score'.format(ti)][()],
                        'feature': fin['/{}/feature'.format(ti)][:],
                        'bboxes': fin['/{}/bboxes'.format(ti)][:]
                    })
        except Exception as e:
            print('[error] failed to open {}'.format(path))
            raise e
    else:
        tracklets = get_object_tracklets(dname, vid, fstart, fend, verbose=verbose)
        num_det = len(tracklets)
        track_gt_id = [-1]*num_det

        # include gt tracklets
        gt_tracklets = {}
        for fid in range(fstart, min(fend, len(anno['trajectories']))):
            rois = anno['trajectories'][fid]
            for roi in rois:
                bbox = [roi['bbox']['xmin'], roi['bbox']['ymin'],
                        roi['bbox']['xmax'], roi['bbox']['ymax']]
                if roi['tid'] not in gt_tracklets:
                    gt_tracklets[roi['tid']] = {
                        'fstart': fid-fstart,
                        'score': 1.0,
                        'bboxes': []
                    }
                n_out_of_view = fid-fstart-gt_tracklets[roi['tid']]['fstart']-len(gt_tracklets[roi['tid']]['bboxes'])
                if n_out_of_view > 0:
                    gt_tracklets[roi['tid']]['bboxes'].extend([_dummy_tiny_bbox]*n_out_of_view)
                gt_tracklets[roi['tid']]['bboxes'].append(bbox)

        for tid, trac in gt_tracklets.items():
            track_gt_id.append(tid)
            trac['bboxes'] = np.asarray(trac['bboxes'], dtype=np.float32)
            tracklets.append(trac)
        track_gt_id = np.asarray(track_gt_id)
        
        pairwise_iou = np.empty((len(tracklets), len(tracklets)), dtype=np.float32)
        for ti, subj in enumerate(tracklets):
            for tj, obj in enumerate(tracklets):
                subj_fstart, subj_fend = subj['fstart'], subj['fstart']+len(subj['bboxes'])
                obj_fstart, obj_fend = obj['fstart'], obj['fstart']+len(obj['bboxes'])
                fstart_min = min(subj_fstart, obj_fstart)
                fend_max = max(subj_fend, obj_fend)
                obj_bboxes = np.pad(obj['bboxes'], ((obj_fstart-fstart_min, fend_max-obj_fend), (0, 0)))
                obj_bboxes = obj_bboxes[(subj_fstart-fstart_min):(subj_fend-fstart_min), :]
                pairwise_iou[ti, tj] = cubic_iou(subj['bboxes'][None, :, :], obj_bboxes[None, :, :])[0, 0]

        # pooling visual feature for gt tracklets
        for ti in range(num_det, len(tracklets)):
            ious = pairwise_iou[:num_det, ti]
            tjs = np.where(ious>0.7)[0]
            if len(tjs) > 0:
                feats = [tracklets[tj]['feature'] for tj in tjs]
                tracklets[ti]['feature'] = np.mean(feats, axis=0)
            else:
                tracklets[ti]['feature'] = np.empty((0,), dtype=np.float32)
        
        with h5py.File(path, 'w') as fout:
            fout.create_dataset('track_gt_id', data=track_gt_id)
            fout.create_dataset('pairwise_iou', data=pairwise_iou)
            for ti, trac in enumerate(tracklets):
                fout.create_dataset('{}/fstart'.format(ti), data=trac['fstart'])
                fout.create_dataset('{}/score'.format(ti), data=trac['score'])
                fout.create_dataset('{}/feature'.format(ti), data=trac['feature'].astype(np.float32))
                fout.create_dataset('{}/bboxes'.format(ti), data=trac['bboxes'])

    if not include_gt:
        num_det = len([i for i in track_gt_id if i < 0])
        tracklets = tracklets[:num_det]
        track_gt_id = track_gt_id[:num_det]
        pairwise_iou = pairwise_iou[:num_det, :]

    return tracklets, track_gt_id, pairwise_iou


def extract_relation_feature(dname, vid, fstart, fend, anno, sampled_pair_ids=None, include_gt=False, verbose=False):
    tracklets, track_gt_id, pairwise_iou = extract_object_feature(dname, vid, fstart, fend, anno, include_gt=include_gt, verbose=verbose)

    num_tracklets = len(tracklets)
    pairs = np.asarray([[ti, tj] for ti, tj in product(range(num_tracklets), range(num_tracklets))])
    if sampled_pair_ids is not None:
        pairs = pairs[sampled_pair_ids]

    sub_feats = []
    obj_feats = []
    pred_pos_feats = []
    pred_vis_feats = []
    for ti, tj in pairs:
        subj = tracklets[ti]
        obj = tracklets[tj]
        subj_fstart, subj_fend = subj['fstart'], subj['fstart']+len(subj['bboxes'])
        obj_fstart, obj_fend = obj['fstart'], obj['fstart']+len(obj['bboxes'])
        start_relative = _compute_relative_positional_feature(subj['bboxes'][0], subj_fstart, obj['bboxes'][0], obj_fstart)
        end_relative = _compute_relative_positional_feature(subj['bboxes'][-1], subj_fend, obj['bboxes'][-1], obj_fend)
        positional_feature = np.concatenate([start_relative, end_relative])
        pred_pos_feats.append(positional_feature.astype(np.float32))

        sub_feats.append(subj['feature'])
        obj_feats.append(obj['feature'])
        pred_vis_feats.append(np.concatenate([subj['feature'], obj['feature']], axis=0))
    sub_feats = np.asarray(sub_feats)
    obj_feats = np.asarray(obj_feats)
    pred_pos_feats = np.asarray(pred_pos_feats)
    pred_vis_feats = np.asarray(pred_vis_feats)
    
    if not include_gt:
        num_det = len([i for i in track_gt_id if i < 0])
        pair_ids = [pair_id for pair_id, (ti, tj) in enumerate(pairs) if track_gt_id[ti] < 0 and track_gt_id[tj] < 0]
        pairs = pairs[pair_ids]
        sub_feats = sub_feats[pair_ids]
        obj_feats = obj_feats[pair_ids]
        pred_pos_feats = pred_pos_feats[pair_ids]
        pred_vis_feats = pred_vis_feats[pair_ids]
        pairwise_iou = pairwise_iou[:num_det, :]

    return pairs, sub_feats, obj_feats, pred_pos_feats, pred_vis_feats, pairwise_iou, track_gt_id


if __name__ == '__main__':
    import argparse
    from dataset import VidVRD, VidOR
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Extract feature and analyze portion of recalled tracklets')
    parser.add_argument('dataset', type=str, help='the dataset name for evaluation')
    parser.add_argument('split', type=str, help='the split name for evaluation')
    args = parser.parse_args()

    if args.dataset=='imagenet-vidvrd':
        dataset = VidVRD('../imagenet-vidvrd-dataset', '../imagenet-vidvrd-dataset/videos', [args.split])
    elif args.dataset=='vidor':
        dataset = VidOR('../vidor-dataset/annotation', '../vidor-dataset/video', [args.split], low_memory=False)
    else:
        raise Exception('Unknown dataset {}'.format(args.dataset))

    indices = []
    video_indices = dataset.get_index(split=args.split)
    for vid in video_indices:
        anno = dataset.get_anno(vid)
        segs = common.segment_video(0, anno['frame_count'])
        for fstart, fend in segs:
            indices.append((vid, fstart, fend))

    num_gt = 0
    num_det = 0
    num_recalls = {0.3: 0, 0.5: 0, 0.7: 0}
    for vid, fstart, fend in tqdm(indices):
        anno = dataset.get_anno(vid)
        pairs, sub_feats, obj_feats, pred_pos_feats, pred_vis_feats, iou, track_gt_id = extract_relation_feature(dataset.name, vid, fstart, fend, anno, include_gt=True)
        if pairs is None:
            continue
        n_gt = len([i for i in track_gt_id if i > -1])
        if n_gt == 0:
            continue
        n_det = len([i for i in track_gt_id if i == -1])
        for thred in num_recalls.keys():
            hits = iou[:n_det, -n_gt:]
            hits = np.any(hits>thred, axis=0)
            n_hit = np.sum(hits)
            num_recalls[thred] += n_hit
        num_gt += n_gt
        num_det += n_det
    
    print('num_gt: {}, num_det: {}'.format(num_gt, num_det))
    for thred, num_recall in num_recalls.items():
        print('num_recall (viou > {}): {:d} ({:.2f}%)'.format(thred, num_recall, num_recall*100.0/num_gt))
