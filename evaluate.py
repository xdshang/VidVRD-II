import os
import json
import argparse
from collections import defaultdict

from tqdm import tqdm

from dataset import VidVRD, VidOR
from evaluation import eval_video_object, eval_action, eval_visual_relation, print_relation_scores
from visualize import visualize


def evaluate_object(dataset, split, prediction):
    groundtruth = dict()
    for vid in dataset.get_index(split):
        groundtruth[vid] = dataset.get_object_insts(vid)
    mean_ap, ap_class = eval_video_object(groundtruth, prediction)


def evaluate_action(dataset, split, prediction):
    groundtruth = dict()
    for vid in dataset.get_index(split):
        groundtruth[vid] = dataset.get_action_insts(vid)
    mean_ap, ap_class = eval_action(groundtruth, prediction)


def evaluate_relation(dataset, split, prediction):
    scores = dict()

    print('[info] evaluating overall setting')
    groundtruth = dict()
    for vid in dataset.get_index(split):
        groundtruth[vid] = dataset.get_relation_insts(vid)
    scores['overall'] = eval_visual_relation(groundtruth, prediction)

    for use_origin_zeroshot_eval in [False, True]:
        if use_origin_zeroshot_eval:
            print('[info] evaluating generalized zero-shot setting')
        else:
            print('[info] evaluating zero-shot setting')
        zeroshot_triplets = dataset.get_triplets(split).difference(
                dataset.get_triplets('train'))
        groundtruth = dict()
        zs_prediction = dict()
        for vid in dataset.get_index(split):
            gt_relations = dataset.get_relation_insts(vid)
            zs_gt_relations = []
            for r in gt_relations:
                if tuple(r['triplet']) in zeroshot_triplets:
                    zs_gt_relations.append(r)
            if len(zs_gt_relations) > 0:
                groundtruth[vid] = zs_gt_relations
                if use_origin_zeroshot_eval:
                    # origin zero-shot evaluation doesn't filter out non-zeroshot predictions
                    # in a video, which is the generalized zero-shot setting 
                    zs_prediction[vid] = prediction[vid]
                else:
                    zs_prediction[vid] = []
                    for r in prediction.get(vid, []):
                        if tuple(r['triplet']) in zeroshot_triplets:
                            zs_prediction[vid].append(r)
        if use_origin_zeroshot_eval:
            scores['generalized zero-shot'] = eval_visual_relation(groundtruth, zs_prediction)
        else:
            scores['zero-shot'] = eval_visual_relation(groundtruth, zs_prediction)

    return scores


def convert_format(anno, pred_relations, pred_version):
    normalize_coords = pred_version >= 'VERSION 2.1'
    entities = []
    trajectories = defaultdict(list)
    relation_instances = []
    if normalize_coords:
        width_ratio = anno['width']
        height_ratio = anno['height']
    else:
        width_ratio = height_ratio = 1

    if pred_version >= 'VERSION 3.0':
        for fid in range(len(pred_relations['trajectories'])):
            frame = pred_relations['trajectories'][fid]
            for region in frame:
                region['bbox']['xmin'] *= width_ratio
                region['bbox']['ymin'] *= height_ratio
                region['bbox']['xmax'] *= width_ratio
                region['bbox']['ymax'] *= height_ratio
            trajectories[fid] = frame
        entities = pred_relations['subject/objects']
        relation_instances = pred_relations['relation_instances']
    else:
        for rel_inst in pred_relations:
            tid = {}
            tid['sub_traj'] = len(entities)
            entities.append({
                'tid': tid['sub_traj'],
                'category': rel_inst['triplet'][0]
            })
            tid['obj_traj'] = len(entities)
            entities.append({
                'tid': tid['obj_traj'],
                'category': rel_inst['triplet'][2]
            })
            fstart, fend = rel_inst['duration']
            relation_instances.append({
                'subject_tid': tid['sub_traj'],
                'object_tid': tid['obj_traj'],
                'predicate': rel_inst['triplet'][1],
                'score': rel_inst['score'],
                'begin_fid': fstart,
                'end_fid': fend
            })
            for e in ['sub_traj', 'obj_traj']:
                for i, bbox in enumerate(rel_inst[e]):
                    trajectories[fstart+i].append({
                        'tid': tid[e],
                        'bbox': {
                            'xmin': bbox[0]*width_ratio,
                            'ymin': bbox[1]*height_ratio,
                            'xmax': bbox[2]*width_ratio,
                            'ymax': bbox[3]*height_ratio
                        }
                    })
    anno = dict(anno)
    anno['subject/objects'] = entities
    anno['trajectories'] = [trajectories[fid] for fid in range(anno['frame_count'])]
    anno['relation_instances'] = relation_instances
    return anno


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a set of tasks related to video relation understanding.')
    parser.add_argument('dataset', type=str, help='the dataset name for evaluation')
    parser.add_argument('split', type=str, help='the split name for evaluation')
    parser.add_argument('task', choices=['object', 'action', 'relation'], help='which task to evaluate')
    parser.add_argument('prediction', type=str, help='Corresponding prediction JSON file')
    parser.add_argument('--visualize', action="store_true", default=False, help='Visualize for qualitative evaluation')
    args = parser.parse_args()

    print('[info] loading prediction from {}'.format(args.prediction))
    with open(args.prediction, 'r') as fin:
        pred = json.load(fin)
    print('------ number of videos in prediction: {}'.format(len(pred['results'])))
    normalize_coords = pred['version'] >= 'VERSION 2.1'

    if args.dataset=='imagenet-vidvrd':
        if args.task=='relation':
            # load train set for zero-shot evaluation
            dataset = VidVRD('../imagenet-vidvrd-dataset', '../imagenet-vidvrd-dataset/videos', ['train', args.split], normalize_coords=normalize_coords)
        else:
            dataset = VidVRD('../imagenet-vidvrd-dataset', '../imagenet-vidvrd-dataset/videos', [args.split], normalize_coords=normalize_coords)
    elif args.dataset=='vidor':
        if args.task=='relation':
            # load train set for zero-shot evaluation
            dataset = VidOR('../vidor-dataset/annotation', '../vidor-dataset/video', ['training', args.split], low_memory=True, normalize_coords=normalize_coords)
        else:
            dataset = VidOR('../vidor-dataset/annotation', '../vidor-dataset/video', [args.split], low_memory=True, normalize_coords=normalize_coords)
    else:
        raise Exception('Unknown dataset {}'.format(args.dataset))

    if args.task=='object':
        assert args.vis_path is None, 'not implemented'
        evaluate_object(dataset, args.split, pred['results'])
    elif args.task=='action':
        assert args.vis_path is None, 'not implemented'
        evaluate_action(dataset, args.split, pred['results'])
    elif args.task=='relation':
        if args.visualize:
            vis_path = os.path.join(os.path.dirname(args.prediction), 'visualize')
            if not os.path.exists(vis_path):
                os.mkdir(vis_path)
            for vid, pred_relations in tqdm(pred['results'].items()):
                anno = dataset.get_anno(vid)
                vis_anno = convert_format(anno, pred_relations, pred['version'])
                video_path = dataset.get_video_path(vid)
                out_path = os.path.join(vis_path, '{}.mp4'.format(vid))
                visualize(vis_anno, video_path, out_path, relation_panel=True, bbox_perturb=0.02)
        else:
            scores = evaluate_relation(dataset, args.split, pred['results'])
            print_relation_scores(scores)
