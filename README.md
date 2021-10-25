## VidVRD-II

This repository contains source codes for "Video Visual Relation Detection via Iterative Inference" (MM'21) [[paper](https://dl.acm.org/doi/abs/10.1145/3474085.3475263)].

### Environment
1. Download [ImageNet-VidVRD dataset](https://xdshang.github.io/docs/imagenet-vidvrd.html) and [VidOR dataset](https://xdshang.github.io/docs/vidor.html). Then, place the data under the same parent folder as this repository (recommended):
```
├── vidor-dataset
│   ├── annotation
│   │   ├── training
│   │   └── validation
│   └── video
├── imagenet-vidvrd-dataset
│   ├── test
│   ├── train
│   └── videos
├── VidVRD-II
│   ├── ... (this repo)
├── vidor-baseline-output
│   ├── ... (intermediate results)
└── imagenet-vidvrd-baseline-output
    ├── ... (intermediate results)
```
2. Install dependencies (tested with TITAN Xp GPU)
<!-- Ubuntu 18.04.5 LTS (GNU/Linux 4.15.0-142-generic x86_64); NVIDIA Driver Version: 460.73.01,  -->
```
conda create -n vidvrd-ii -c conda-forge python=3.7 Cython tqdm scipy "h5py>=2.9=mpi*" ffmpeg=3.4 cudatoolkit=10.1 cudnn "pytorch>=1.7.0=cuda101*" "tensorflow>=2.0.0=gpu*"
conda activate vidvrd-ii
python setup.py build_ext --inplace
``` 
3. [Optional] Since `cv2` is incompatible with the main environment in vidvrd-ii, if you want to use the script `visualize.py`, you may need to create a separate environment with `py-opencv` installed.

### Quick Start
1. Download the precomputed object tracklets and features for ImageNet-VidVRD ([437MB](https://zdtnag7mmr.larksuite.com/file/boxusOKDuonKBcA2Le9JMeu3ePg)) and VidOR (32GB: [part1](https://zdtnag7mmr.larksuite.com/file/boxusff3nUWfZly119sJXhLkpWe), [part2](https://zdtnag7mmr.larksuite.com/file/boxus2FJ47VRRNSUZ4f9aJzICAc), [part3](https://zdtnag7mmr.larksuite.com/file/boxusr0Bb6cSX4OQXCjKvhHf94d), [part4](https://zdtnag7mmr.larksuite.com/file/boxusGIpO3VDPkfNTtzUqMRIdrc)), and extract them under `imagenet-vidvrd-baseline-output` and `vidor-baseline-output` as above, respectively.
2. Run `python main.py --cfg config/imagenet_vidvrd_3step_prop_wd0.01.json --id 3step_prop_wd0.01 --train --cuda` to train the model for ImageNet-VidVRD. Use `--cfg config/vidor_3step_prop_wd1.json` for VidOR.
3. Run `python main.py --cfg config/imagenet_vidvrd_3step_prop_wd0.01.json --id 3step_prop_wd0.01 --detect --cuda` to detect video relations (inference) and the results will be output to `../imagenet-vidvrd-baseline-output/models/3step_prop_wd0.01/video_relations.json`.
4. Run `python evaluate.py imagenet-vidvrd test relation ../imagenet-vidvrd-baseline-output/models/3step_prop_wd0.01/video_relations.json` to evaluate the results.
5. To visualize the results, add the option `--visualize` to the above command (this will involve `visualize.py` so please make sure the environment is switched according to the last section). For the better visualization mentioned in the paper, change `association_algorithm` to `graph` in the configuration json, and then run Step 3 and 5.
6. To automatically run the whole traininng and test pipepine multiple times, run `python main.py --cfg config/imagenet_vidvrd_3step_prop_wd0.01.json --id 3step_prop_wd0.01 --pipeline 5 --cuda --no_cache` and then you can obtain a mean/std result.

### Object Tracklet Extraction (optional)
1. We extract frame-level object proposals using the off-the-shelf tool. Please first download and install [tensorflow model library](https://github.com/tensorflow/models/tree/master/research/object_detection). Then, run `python -m video_object_detection.tfmodel_image_detection [imagenet-vidvrd/vidor] [train/test/training/validation]`. You can also download our precomputed results for ImageNet-VidVRD ([6GB](https://zdtnag7mmr.larksuite.com/file/boxuspVnSh0mnRW4Zdxh3282w4d)).
2. To obtain object tracklets based on the frame-level proposals, run `python -m video_object_detection.object_tracklet_proposal [imagenet-vidvrd/vidor] [train/test/training/validation]`.

### Acknowledgement
This repository is built based on [VidVRD-helper](https://github.com/xdshang/VidVRD-helper). If this repo is helpful in your research, you can use the following bibtex to cite the paper:
```
@inproceedings{shang2021video,
  title={Video Visual Relation Detection via Iterative Inference},
  author={Shang, Xindi and Li, Yicong and Xiao, Junbin and Ji, Wei and Chua, Tat-Seng},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={3654--3663},
  year={2021}
}
```
