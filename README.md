# Cross-Iteration Batch Normalization
This repository contains a PyTorch implementation of the CBN layer, as well as some training scripts to reproduce the COCO object detection and instance segmentation results reported in our paper.


## Results with this code

| Backbone      | Method       | Norm | AP<sup>b</sup> | AP<sup>b</sup><sub>0.50</sub> | AP<sup>b</sup><sub>0.75</sub> | AP<sup>m</sup> | AP<sup>m</sup><sub>0.50</sub> | AP<sup>m</sup><sub>0.75</sub> | Download |
|:-------------:|:------------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| R-50-FPN | Faster R-CNN | -      | 36.8 | 57.9 | 39.8 | - | - | - | model |
| R-50-FPN | Faster R-CNN | SyncBN | 37.5 | 58.4 | 40.6 | - | - | - | model |
| R-50-FPN | Faster R-CNN | GN     | 37.7 | 59.2 | 41.2 | - | - | - | model |
| R-50-FPN | Faster R-CNN | CBN    | 37.6 | 58.5 | 40.9 | - | - | - | model |
| R-50-FPN | Mask R-CNN | -      | 37.6 | 58.5 | 41.0 | 34.0 | 55.2 | 36.2 | model |
| R-50-FPN | Mask R-CNN | SyncBN | 38.5 | 58.9 | 42.0 | 34.3 | 55.7 | 36.7 | model |
| R-50-FPN | Mask R-CNN | GN     | 38.5 | 59.4 | 41.8 | 35.0 | 56.4 | 37.3 | model |
| R-50-FPN | Mask R-CNN | CBN    | 38.4 | 58.9 | 42.2 | 34.7 | 55.9 | 37.0 | model |

*All results are trained with 1x schedule. Normalization layers of backbone are fixed by default.


## Installation
Please refer to [INSTALL.md](INSTALL.md) for installation and dataset preparation.


## Demo

### Test
Download the pretrained model
```bash
# Faster R-CNN
python tools/test.py {configs_file} {downloaded model} --gpus 4 --out {tmp.pkl} --eval bbox
# Mask R-CNN
python tools/test.py {configs_file} {downloaded model} --gpus 4 --out {tmp.pkl} --eval bbox segm
```


### Train Mask R-CNNN
One node with 4GPUs:
```bash
# SyncBN
./tools/dist_train.sh ./configs/cbn/mask_rcnn_r50_fpn_syncbn_1x.py 4
# GN
./tools/dist_train.sh ./configs/cbn/mask_rcnn_r50_fpn_gn_1x.py 4
# CBN
./tools/dist_train.sh ./configs/cbn/mask_rcnn_r50_fpn_cbn_buffer3_burnin8_1x.py 4
```


## TODO
- [x] Clean up mmdetection code base
- [x] Add CBN layer support
- [x] Add default configs for training
- [x] Upload pretrained models for quick test demo
- [ ] Provide a conv_module of Conv & CBN
- [ ] Speedup CBN layer with CUDA/CUDNN


## Thanks
This implementation is based on [mmdetection](https://github.com/open-mmlab/mmdetection). Ref to this link for more details about [mmdetection](https://github.com/open-mmlab/mmdetection).


## Citation
If you use Cross-Iteration Batch Normalization in your research, please cite:
```bibtex
@inproceedings{
  anonymous2020crossiteration,
  title={Cross-Iteration Batch Normalization},
  author={Anonymous},
  booktitle={Submitted to International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=BylJUTEKvB},
  note={under review}
}
```
