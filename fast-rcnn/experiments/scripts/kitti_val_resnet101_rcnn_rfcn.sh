#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/kitti_val_resnet101_rcnn_rfcn.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
 --solver models/ResNet101/kitti_val/solver_rcnn_rfcn_7x7.prototxt \
 --weights data/imagenet_models/ResNet-101-model.caffemodel \
 --imdb kitti_train\
 --iters 40000\
 --cfg experiments/cfgs/kitti_rcnn_vgg16_multiscale_1d0.yml \

time ./tools/test_net.py --gpu $1 \
  --def models/ResNet101/kitti_val/test_rcnn_rfcn_7x7.prototxt \
  --net output/kitti/kitti_train/resnet101_fast_rcnn_rfcn_7x7_kitti_iter_40000.caffemodel \
  --imdb kitti_val \
  --cfg experiments/cfgs/kitti_rcnn_vgg16_multiscale_1d0.yml








