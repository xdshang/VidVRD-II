import os
import json
import pickle
import argparse
import random
import multiprocessing
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import torch

from dataset import VidVRD, VidOR
import common
from common import association
from evaluate import evaluate_relation, print_relation_scores


def get_model_param(cfg_path, exp_id, no_cache=False):
    with open(cfg_path, 'r') as fin:
        param = json.load(fin)
    param['exp_id'] = exp_id

    saved_param_path = os.path.join(common.get_model_path(exp_id, param['dataset']), 'setting.json')
    if not no_cache and os.path.exists(saved_param_path):
        with open(saved_param_path, 'r') as fin:
            param.update(json.load(fin))
            assert param['exp_id'] == exp_id
        with open(cfg_path, 'r') as fin:
            param.update(json.load(fin))
    
    torch.manual_seed(param['rng_seed'])
    np.random.seed(param['rng_seed'])
    random.seed(param['rng_seed'])
    torch.cuda.manual_seed_all(param['rng_seed'])

    if param['model']['name'] in ['independent_classifier', 'cascade_classifier', 'iterative_classifier']:
        from baseline import predictor
        from baseline import learner
    else:
        raise ValueError(param['model']['name'])

    print(json.dumps(param, indent=4))
    return param, predictor, learner


def train(dataset, param, learner, use_cuda=False, no_cache=False):
    param['use_cached_training_sample'] = (not no_cache)
    param = learner.train(dataset, param['train_split'], use_cuda=use_cuda, **param)
    with open(os.path.join(common.get_model_path(param['exp_id'], param['dataset']), 'setting.json'), 'w') as fout:
        json.dump(param, fout, indent=4)
    return param


def eval_relation_segments(dataset, param, predictor, use_cuda=False, no_cache=False):
    res_path = os.path.join(common.get_model_path(param['exp_id'], param['dataset']), 'relation_segments.pkl')
    if not no_cache and os.path.exists(res_path):
        with open(res_path, 'rb') as fin:
            relation_segments = pickle.load(fin)['results']
    else:
        relation_segments = predictor.predict(dataset, param['test_split'], use_cuda=use_cuda, **param)
        with open(res_path, 'wb') as fout:
            pickle.dump({
                'version': 'VERSION 2.1',
                'results': relation_segments
            }, fout)

    common.eval_relation_segments(dataset, dataset.get_index(split=param['test_split']), relation_segments)


def detect(dataset, param, predictor, use_cuda=False, no_cache=False, n_workers=12):
    res_path = os.path.join(common.get_model_path(param['exp_id'], param['dataset']), 'relation_segments.pkl')
    if not no_cache and os.path.exists(res_path):
        with open(res_path, 'rb') as fin:
            _data = pickle.load(fin)
            version = _data['version']
            relation_segments = _data['results']
    else:
        version = 'VERSION 2.1'
        relation_segments = predictor.predict(dataset, param['test_split'], use_cuda=use_cuda, **param)
        with open(res_path, 'wb') as fout:
            pickle.dump({
                'version': version,
                'results': relation_segments
            }, fout)

    # group relation segments by video
    test_indices = dataset.get_index(split=param['test_split'])
    segment_groups = {}
    for vid in test_indices:
        segment_groups[vid] = {}
        anno = dataset.get_anno(vid)
        segs = common.segment_video(0, anno['frame_count'])
        for fstart, fend in segs:
            vsig = common.get_segment_signature(vid, fstart, fend)
            segment_groups[vid][(fstart, fend)] = relation_segments.get(vsig, [])

    # video-level visual relation detection by relational association
    print('[info] {} relation association using {} workers'.format(param['association_algorithm'], param['association_n_workers']))
    if param['association_algorithm'] == 'greedy':
        algorithm = association.greedy_relation_association
    elif param['association_algorithm'] == 'nms':
        algorithm = association.nms_relation_association
    elif param['association_algorithm'] == 'graph':
        algorithm = association.greedy_graph_association
        version = 'VERSION 3.0'
    else:
        raise ValueError(param['association_algorithm'])

    video_relations = {}
    if param.get('association_n_workers', 0) > 0:
        with tqdm(total=len(test_indices)) as pbar:
            pool = multiprocessing.Pool(processes=param['association_n_workers'])
            for vid in test_indices:
                video_relations[vid] = pool.apply_async(association.parallel_association,
                        args=(vid, algorithm, segment_groups[vid], param),
                        callback=lambda _: pbar.update())
            pool.close()
            pool.join()
        for vid in video_relations.keys():
            res = video_relations[vid].get()
            video_relations[vid] = res
    else:
        for vid in tqdm(test_indices):
            res = algorithm(segment_groups[vid], **param)
            video_relations[vid] = res

    return {'version': version, 'results': video_relations}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VidVRD main script')
    parser.add_argument('--cfg', type=str, required=True, help='Path to the config file')
    parser.add_argument('--id', type=str, required=True, help='Experiment ID')
    parser.add_argument('--train', action="store_true", default=False, help='Train model')
    parser.add_argument('--eval_relation_segment', action="store_true", default=False, help='Evaluate relation segments')
    parser.add_argument('--detect', action="store_true", default=False, help='Detect video visual relation')
    parser.add_argument('--pipeline_train', type=int, help='Run train pipeline N times')
    parser.add_argument('--pipeline', type=int, help='Run train/test pipeline N times and output average evaluation results')
    parser.add_argument('--cuda', action="store_true", default=False, help='Use GPU')
    parser.add_argument('--rpath', default='../', help='Root path to store intermediate results')
    parser.add_argument('--no_cache', action='store_true', help='Do not use cached intermediate results')
    args = parser.parse_args()

    common.misc.rpath = args.rpath
    param, predictor, learner = get_model_param(args.cfg, args.id, no_cache=args.train)

    splits = [param['train_split'], param['test_split']]
    if param['dataset'] == 'imagenet-vidvrd':
        dataset = VidVRD('../imagenet-vidvrd-dataset', '../imagenet-vidvrd-dataset/videos', splits)
    elif param['dataset'] == 'vidor':
        dataset = VidOR('../vidor-dataset/annotation', '../vidor-dataset/video', splits, low_memory=False)
    else:
        raise Exception('Unknown dataset {}'.format(param['dataset']))

    if args.train:
        train(dataset, param, learner, args.cuda, args.no_cache)

    elif args.eval_relation_segment:
        eval_relation_segments(dataset, param, predictor, args.cuda, args.no_cache)

    elif args.detect:
        output = detect(dataset, param, predictor, args.cuda, args.no_cache)
        output_path = os.path.join(common.get_model_path(param['exp_id'], param['dataset']), 'video_relations.json')
        print('[info] saving video relations to {}'.format(output_path))
        with open(output_path, 'w') as fout:
            json.dump(output, fout)

    elif args.pipeline_train:
        for n in range(args.pipeline_train):
            print('\nPipeline Run {}'.format(n+1))
            print('='*120)
            if n > 0:
                args.no_cache = False
            if args.pipeline_train > 1:
                param['model']['dump_file_prefix'] = 'run{}_'.format(n+1)
            model_path = os.path.join(common.get_model_path(param['exp_id'], param['dataset']), 'weights',
                    '{}{}'.format(param['model'].get('dump_file_prefix', ''), param['model']['dump_file']))
            if not os.path.exists(model_path):
                param = train(dataset, param, learner, args.cuda, args.no_cache)
            else:
                print('[info] found trained model: {}'.format(model_path))

    elif args.pipeline:
        all_scores = dict()
        for n in range(args.pipeline):
            print('\nPipeline Run {}'.format(n+1))
            print('='*120)
            if n > 0:
                args.no_cache = False
            if args.pipeline > 1:
                param['model']['dump_file_prefix'] = 'run{}_'.format(n+1)
            model_path = os.path.join(common.get_model_path(param['exp_id'], param['dataset']), 'weights',
                    '{}{}'.format(param['model'].get('dump_file_prefix', ''), param['model']['dump_file']))
            if not os.path.exists(model_path):
                param = train(dataset, param, learner, args.cuda, args.no_cache)
            output = detect(dataset, param, predictor, args.cuda, no_cache=True)
            scores = evaluate_relation(dataset, param['test_split'], output['results'])
            print_relation_scores(scores)
            for setting in scores.keys():
                if setting not in all_scores:
                    all_scores[setting] = defaultdict(list)
                for metric in scores[setting].keys():
                    all_scores[setting][metric].append(scores[setting][metric])

        print('\nMean/Std Scores over {} Runs'.format(args.pipeline))
        print('='*120)
        mean_scores = defaultdict(dict)
        std_scores = defaultdict(dict)
        for setting in all_scores.keys():
            for metric in all_scores[setting].keys():
                mean_scores[setting][metric] = np.mean(all_scores[setting][metric])
                std_scores[setting][metric] = np.std(all_scores[setting][metric])

        print_relation_scores(mean_scores, score_variance=std_scores)

    else:
        parser.print_help()
