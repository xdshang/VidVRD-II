#!/usr/bin/env python
# coding: utf-8
"""
Adapted from https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
Using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions]
(https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.
"""

import numpy as np
import os
import sys
import math
import glob
import tempfile
import subprocess
import argparse
import tensorflow as tf
from tf.keras.preprocessing.image import load_img, img_to_array, save_img
from distutils.version import StrictVersion
if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

from tqdm import tqdm
import h5py

import common
from dataset import VidVRD, VidOR

sys.path.append(common.get_tensorflow_research_model_path())
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def visualize_frame(image_np, bboxes, autotags, scores, category_index, save_path):
    image_np = np.array(image_np)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
            image_np, bboxes, autotags, scores, category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
    vis_vpath = os.path.dirname(save_path)
    if not os.path.exists(vis_vpath):
        os.makedirs(vis_vpath)
    save_img(save_path, image_np)


def run_inference_for_single_video(sess, input_holder, category_index, dataset, vid, batch_size=1, ffmpeg_cmd='ffmpeg', visualize=False):
    predictor_name = 'tensorflow_faster_rcnn_oiv4'
    output_path = common.get_object_detection_path(dataset.name, 'frame_{}'.format(predictor_name), '{}.h5'.format(vid))
    if os.path.exists(output_path):
        try:
            with h5py.File(output_path, 'r') as fin:
                pass
            return
        except OSError:
            print('[error] file {} broken (processing again)'.format(output_path))

    vpath = dataset.get_video_path(vid)
    with tempfile.TemporaryDirectory(dir='/dev/shm') as frame_dir:
        subprocess.run([ffmpeg_cmd, '-i', vpath, '{}/%05d.JPEG'.format(frame_dir)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        frame_paths = glob.glob('{}/*'.format(frame_dir))
        frame_count = dataset.get_anno(vid)['frame_count']
        if frame_count > len(frame_paths):
            print('[warning] video {} cannot extract {} frames (extracted {} frames)'.format(vid, frame_count, len(frame_paths))) 
        
        with h5py.File(output_path, 'w') as fout:
            for i in range(0, len(frame_paths), batch_size):
                paths = frame_paths[i: i+batch_size]
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = []
                for image_path in paths:
                    image_np = img_to_array(load_img(image_path))
                    image_np_expanded.append(image_np)
                image_np_expanded = np.asarray(image_np_expanded)
                if image_np_expanded.shape[0]<batch_size:
                    npad = (0, batch_size-image_np_expanded.shape[0]), (0, 0), (0, 0), (0, 0)
                    image_np_expanded = np.pad(image_np_expanded, pad_width=npad, mode='constant', constant_values=0.)
                # Run inference
                output_dict = sess.run(tensor_dict, feed_dict={input_holder: image_np_expanded})
                dims = output_dict['detection_scores'].shape+(-1,)
                output_dict['SecondStagePostprocessor/Softmax'] = output_dict['SecondStagePostprocessor/Softmax'].reshape(dims)
                output_dict['SecondStageBoxPredictor/Flatten_1/flatten/Reshape'] = \
                        output_dict['SecondStageBoxPredictor/Flatten_1/flatten/Reshape'].reshape(dims)
                # decode the inference results
                for j, image_path in enumerate(paths):
                    dets = dict()
                    # all outputs are float32 numpy arrays, so convert types as appropriate
                    dets['num_detections'] = int(output_dict['num_detections'][j])
                    dets['detection_classes'] = output_dict[
                            'detection_classes'][j].astype(np.int64)[:dets['num_detections']]
                    dets['detection_boxes'] = output_dict['detection_boxes'][j][:dets['num_detections']]
                    dets['detection_scores'] = output_dict['detection_scores'][j][:dets['num_detections']]
                    dets['classmes'] = output_dict['SecondStagePostprocessor/Softmax'][j]
                    dets['features'] = output_dict['SecondStageBoxPredictor/Flatten_1/flatten/Reshape'][j]

                    fid = int(os.path.splitext(os.path.basename(image_path))[0])
                    bboxes = []
                    autotags = []
                    scores = []
                    classmes = []
                    features = []
                    for box, cid, score in zip(dets['detection_boxes'], dets['detection_classes'], dets['detection_scores']):
                        # we are only interested in a subset of detected objects by the pretrained model
                        if cid in category_index:
                            # find corresponding classeme and feature according to the score
                            indice, _ = np.where(score==dets['classmes'][:, 1:])
                            if indice.size==0:
                                print('[warning] couldn\'t find corresponding classeme vector in {}th frame of {}'.format(fid, vid))
                            elif indice.size>1:
                                print('[warning] found multiple classme vector in {}th frame of {}'.format(fid, vid))
                            else:
                                bboxes.append(box)
                                autotags.append(cid)
                                scores.append(score)
                                classmes.append(dets['classmes'][indice[0]])
                                features.append(dets['features'][indice[0]])

                    fout.create_dataset('{}/bboxes'.format(fid), data=bboxes)
                    fout.create_dataset('{}/autotags'.format(fid),
                            data=np.asarray([category_index[cid]['name'] for cid in autotags], dtype='S10'))
                    fout.create_dataset('{}/scores'.format(fid), data=scores)
                    fout.create_dataset('{}/classmes'.format(fid), data=classmes)
                    fout.create_dataset('{}/features'.format(fid), data=features)

                    if visualize:
                        save_path = os.path.join(common.get_object_detection_path('{}_frame'.format(dataset.name), vid), 
                                os.path.basename(image_path))
                        visualize_frame(image_np_expanded[j], np.asarray(bboxes), np.asarray(autotags), np.asarray(scores),
                                category_index, save_path)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run image object detection over the given dataset')
    parser.add_argument('dataset', type=str, help='the dataset name for evaluation')
    parser.add_argument('split', type=str, help='the split name for evaluation')
    parser.add_argument('--parallel', type=str, help='Indicate the data parallel argument, e.g. "3/8" and "8/8"')
    parser.add_argument('--batchsize', type=int, default=1, help='Batch size for model inference. 1 if GPU memory is less than 12GB')
    parser.add_argument('--ffmpeg', type=str, default='ffmpeg', help='Path to customized ffmpeg binary')
    args = parser.parse_args()

    if args.dataset=='vidvrd':
        dataset = VidVRD('../imagenet-vidvrd-dataset', '../imagenet-vidvrd-dataset/videos', [args.split])
    elif args.dataset=='vidor':
        dataset = VidOR('../vidor-dataset/annotation', '../vidor-dataset/video', [args.split], low_memory=True)
    else:
        raise Exception('Unknown dataset {}'.format(args.dataset))

    print('Intializing...') 
    """
    Load a (frozen) Tensorflow model into memory. Download the model from
    http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12.tar.gz
    """
    PATH_TO_FROZEN_GRAPH = os.path.join(get_tensorflow_research_model_path(), 'object_detection',
            'faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12', 'frozen_inference_graph.pb')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    """
    Label maps map indices to category names, so that when our convolution network predicts `5`, 
    we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything 
    that returns a dictionary mapping integers to appropriate string labels would be fine
    List of the strings that is used to add correct label for each box.
    """
    category_index = label_map_util.create_category_index_from_labelmap('video_object_detection/oid_v4_vidvrd_label_map.pbtxt', use_display_name=True)

    with detection_graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes',
                    'SecondStagePostprocessor/Softmax',
                    'SecondStageBoxPredictor/Flatten_1/flatten/Reshape']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            if args.parallel:
                gid, num_grp = args.parallel.split('/')
                gid = int(gid)
                num_grp = int(num_grp)
                video_indices = sorted(dataset.get_index(split=args.split))
                num_video = len(video_indices)
                grp_size = math.ceil(num_video/num_grp)
                video_indices = video_indices[grp_size*(gid-1):grp_size*gid]
                print('[info] processing {}th to {}th video from {} videos...'.format(grp_size*(gid-1)+1, grp_size*gid, num_video))
            else:
                video_indices = dataset.get_index(split=args.split)
            for vid in tqdm(video_indices):
                run_inference_for_single_video(sess, image_tensor, category_index, dataset, vid, batch_size=args.batchsize, ffmpeg_cmd=args.ffmpeg)
