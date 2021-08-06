import os
import argparse
from collections import defaultdict

import h5py
from tqdm import tqdm
import numpy as np

import common
from video_object_detection.seq_nms import seq_nms
from dataset import VidVRD, VidOR
from evaluation import eval_video_object
from visualize import draw_anno, read_video, write_video


def visualize_object_tracklets(dataset, vid, fstart, fend, tracklets):
    anno = dataset.get_anno(vid)
    video_path = dataset.get_video_path(vid)
    video = read_video(video_path)
    assert anno['frame_count']<=len(video), '{} : anno {} video {}'.format(anno['video_id'], anno['frame_count'], len(video))
    assert anno['width']==video[0].shape[1] and anno['height']==video[0].shape[0],\
            '{} : anno ({}, {}) video {}'.format(anno['video_id'], anno['height'], anno['width'], video[0].shape)
    video = video[fstart: fend]

    info = {
        'height': anno['height'],
        'width': anno['width'],
        'frame_count': fend-fstart,
        'subject/objects': [],
        'trajectories': [[] for _ in range(fend-fstart)],
        'relation_instances': []
    }
    for i, trac in enumerate(tracklets):
        info['subject/objects'].append({
            'tid': i,
            'category': 'object'
        })
        for bi, bbox in enumerate(trac['bboxes']):
            info['trajectories'][trac['fstart']+bi].append({
                'tid': i,
                'bbox': {
                    'xmin': bbox[0]*anno['width'],
                    'ymin': bbox[1]*anno['height'],
                    'xmax': bbox[2]*anno['width'],
                    'ymax': bbox[3]*anno['height']
                }
            })
    video, size = draw_anno(video, info)
    vsig = common.get_segment_signature(vid, fstart, fend)
    out_path = os.path.join(common.get_object_detection_path(dataset.name, 'clip_vis', vid), '{}.mp4'.format(vsig))
    write_video(video, anno['fps']/2, size, out_path)   


def get_object_tracklets(dname, vid, fstart, fend, tolerance=30,
        scale_ratio_threshold=2.0, area_ratio_threshold=2.0, linkage_threshold=0.5, nms_threshold=0.7,
        tag_constrained=True, verbose=False):
    vsig = common.get_segment_signature(vid, fstart, fend)
    output_path = os.path.join(common.get_object_detection_path(dname, 'clip', vid), '{}.h5'.format(vsig))
    if os.path.exists(output_path):
        if verbose:
            print('[info] loading existing {}'.format(output_path))
        try:
            fin = h5py.File(output_path, 'r')
        except Exception as e:
            print('[error] failed to open {}'.format(output_path))
            raise e
        tracklets = []
        for i in fin['/'].keys():
            tracklets.append({
                'fstart': fin['/{}/fstart'.format(i)][()],
                'score': fin['/{}/score'.format(i)][()],
                'feature': fin['/{}/feature'.format(i)][:],
                'bboxes': fin['/{}/bboxes'.format(i)][:]
            })
        fin.close()
    else:
        input_path = common.get_object_detection_path(dname, 'frame', '{}.h5'.format(vid))
        raw_bboxes = []
        raw_scores = []
        raw_features = []
        raw_tags = []
        max_det_num = 0
        with h5py.File(input_path, 'r') as fin:
            for fid in range(fstart, fend):
                if str(fid) in fin['/'].keys():
                    bboxes = fin['/{}/bboxes'.format(fid)][:]
                    scores = fin['/{}/scores'.format(fid)][:]
                    features = fin['/{}/features'.format(fid)][:]
                    autotags = [t.decode() for t in fin['/{}/autotags'.format(fid)]]
                if str(fid) not in fin['/'].keys() or bboxes.shape[0] == 0:
                    bboxes = np.zeros((1, 4), dtype=np.float32)
                    scores = np.asarray([-float('inf')], dtype=np.float32)
                    features = np.zeros((0,), dtype=np.float32)
                    autotags = ['N.A.']
                else:
                    bboxes = bboxes[:, [1, 0, 3, 2]]
                max_det_num = max(bboxes.shape[0], max_det_num)
                raw_bboxes.append(bboxes)
                raw_scores.append(scores)
                raw_features.append(features)
                raw_tags.append(autotags)
        if max_det_num > 0:
            # Improved Seq-NMS according to Xie et al.
            # "Video Relation Detection with Trajectory-aware Multi-modalFeatures" ACM MM'20
            padded_bboxes = defaultdict(list)
            padded_scores = defaultdict(list)
            padded_tags = defaultdict(list)
            for i in range(len(raw_bboxes)):
                for ni in range(raw_bboxes[i].shape[0]):
                    if raw_tags[i][ni] == 'N.A.':
                        continue
                    for j in range(i+2, min(i+tolerance, len(raw_bboxes))):
                        found = False
                        for nj in range(raw_bboxes[j].shape[0]):
                            if (tag_constrained and raw_tags[i][ni] != raw_tags[j][nj]) or raw_tags[j][nj] == 'N.A.':
                                continue
                            wi = raw_bboxes[i][ni, 2]-raw_bboxes[i][ni, 0]
                            hi = raw_bboxes[i][ni, 3]-raw_bboxes[i][ni, 1]
                            wj = raw_bboxes[j][nj, 2]-raw_bboxes[j][nj, 0]
                            hj = raw_bboxes[j][nj, 3]-raw_bboxes[j][nj, 1]
                            scale_ratio = max(wi*hj/(wj*hi), wj*hi/(wi*hj))
                            area_ratio = max(wi*hi/(wj*hj), wj*wj/(wi*hi))
                            if scale_ratio > scale_ratio_threshold or area_ratio > area_ratio_threshold:
                                continue
                            for k in range(i+1, j):
                                interpolated = list(map(lambda c: np.interp(k, [i, j], [raw_bboxes[i][ni, c], raw_bboxes[j][nj, c]]), range(4)))
                                padded_bboxes[k].append(interpolated)
                                padded_scores[k].append(0.)
                                padded_tags[k].append(raw_tags[i][ni])
                            found = True
                        if found:
                            break
            for k in padded_bboxes.keys():
                raw_bboxes[k] = np.append(raw_bboxes[k], np.asarray(padded_bboxes[k], dtype=np.float32), axis=0)
                raw_scores[k] = np.append(raw_scores[k], np.asarray(padded_scores[k], dtype=np.float32), axis=0)
                raw_tags[k].extend(padded_tags[k])
                max_det_num = max(raw_bboxes[k].shape[0], max_det_num)

            padded_bboxes = []
            padded_scores = []
            for bboxes in raw_bboxes:
                padded_bboxes.append(np.pad(bboxes, ((0, max_det_num-bboxes.shape[0]), (0, 0)), mode='constant', constant_values=0))
            for scores in raw_scores:
                padded_scores.append(np.pad(scores, (0, max_det_num-scores.shape[0]), mode='constant', constant_values=-float('inf')))
            padded_bboxes = np.stack(padded_bboxes)
            padded_scores = np.stack(padded_scores)
            sequences = seq_nms(padded_bboxes, padded_scores, raw_tags if tag_constrained else [], 
                    linkage_threshold=linkage_threshold, nms_threshold=nms_threshold)
            tracklets = []
            for fstart, sequence, score in sequences:
                bboxes = np.stack([padded_bboxes[fstart+i, bid] for i, bid in enumerate(sequence)])
                feature = np.mean([raw_features[fstart+i][bid] for i, bid in enumerate(sequence) 
                        if bid < raw_features[fstart+i].shape[0]], axis=0)
                tracklets.append({
                    'fstart': fstart,
                    'score': score,
                    'feature': feature,
                    'bboxes': bboxes
                })
        else:
            tracklets = []

        with h5py.File(output_path, 'w') as fout:
            for i, trac in enumerate(tracklets):
                fout.create_dataset('{}/fstart'.format(i), data=trac['fstart'])
                fout.create_dataset('{}/score'.format(i), data=trac['score'])
                fout.create_dataset('{}/feature'.format(i), data=trac['feature'])
                fout.create_dataset('{}/bboxes'.format(i), data=trac['bboxes'])

    return tracklets


def eval_object_tracklets(dataset, split, class_agnostic=True, visualize=False):
    groundtruth = dict()
    prediction = dict()

    segment_indices = []
    video_indices = dataset.get_index(split=split)
    print('[info] generating ground truth object tracklets')
    for vid in tqdm(video_indices):
        anno = dataset.get_anno(vid)
        objects = dataset.get_object_insts(vid)
        segs = common.segment_video(0, anno['frame_count'])
        for fstart, fend in segs:
            segment_indices.append((vid, fstart, fend))
            vsig = common.get_segment_signature(vid, fstart, fend)
            tracklets = []
            for obj in objects:
                rois = {}
                for fid, roi in obj['trajectory'].items():
                    if fstart <= int(fid) < fend:
                        rois[str(int(fid)-fstart)] = roi
                if len(rois) > 0:
                    tracklets.append({
                        'category': 'object' if class_agnostic else obj['category'],
                        'trajectory': rois
                    })
            groundtruth[vsig] = tracklets

    print('[info] generating object tracklets from image object detections')
    num_dets = []
    for vid, fstart, fend in tqdm(segment_indices):
        vsig = common.get_segment_signature(vid, fstart, fend)
        anno = dataset.get_anno(vid)
        # get predicted tracklets
        _tracklets = get_object_tracklets(dataset.name, vid, fstart, fend)
        tracklets = []
        for _trac in _tracklets:
            trac = {
                'category': 'object' if class_agnostic else _trac['category'],
                'score': _trac['score'],
                'trajectory': dict((str(_trac['fstart']+i), bbox) for i, bbox in enumerate(_trac['bboxes']))
            }
            tracklets.append(trac)
        num_dets.append(len(tracklets))
        prediction[vsig] = tracklets
        if visualize:
            visualize_object_tracklets(dataset, vid, fstart, fend, _tracklets)

    print('[info] mean/median/max number of detect object tracklets: {}, {}, {}'.format(
        np.mean(num_dets), np.median(num_dets), np.max(num_dets)))
    mean_ap, ap_class = eval_video_object(groundtruth, prediction, thresh_t=0.5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run object tracklet proposal over the given dataset')
    parser.add_argument('dataset', type=str, help='the dataset name for evaluation')
    parser.add_argument('split', type=str, help='the split name for evaluation')
    parser.add_argument('--visualize', action='store_true', help='whether to visualize')
    args = parser.parse_args()


    if args.dataset=='imagenet-vidvrd':
        dataset = VidVRD('../imagenet-vidvrd-dataset', '../imagenet-vidvrd-dataset/videos', [args.split])
    elif args.dataset=='vidor':
        dataset = VidOR('../vidor-dataset/annotation', '../vidor-dataset/video', [args.split], low_memory=False)
    else:
        raise Exception('Unknown dataset {}'.format(args.dataset))

    eval_object_tracklets(dataset, args.split, visualize=args.visualize)
