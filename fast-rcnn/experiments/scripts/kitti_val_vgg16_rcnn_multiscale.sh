#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/kitti_val_vgg16_rcnn_multiscale_rfcn_2d5_RPNhem.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

#time ./tools/train_net.py --gpu $1 \
#  --solver models/VGG16/kitti_val/solver_rcnn_multiscale_rfcn.prototxt \
#  --weights data/imagenet_models/VGG16.v2.caffemodel \
#  --imdb kitti_train\
#  --iters 80000\
#  --cfg experiments/cfgs/kitti_rcnn_vgg16_multiscale_2d5_nosub.yml \

# time ./tools/train_net.py --gpu $1 \
#   --solver models/VGG16/kitti_val/solver_rcnn_multiscale_rfcn_2d0_5x5.prototxt \
#   --weights data/imagenet_models/VGG16.v2.caffemodel \
#   --imdb kitti_train\
#   --iters 80000\
#   --cfg experiments/cfgs/kitti_rcnn_vgg16_multiscale_2d0.yml \

# time ./tools/train_net.py --gpu $1 \
#   --solver models/VGG16/kitti_val/solver_rcnn_multiscale_rfcn_2d5_3x3.prototxt \
#   --weights data/imagenet_models/VGG16.v2.caffemodel \
#   --imdb kitti_train\
#   --iters 80000\
#   --cfg experiments/cfgs/kitti_rcnn_vgg16_multiscale_2d5.yml \

#time ./tools/train_net.py --gpu $1 \
#  --solver models/VGG16/kitti_val/solver_rcnn_multiscale_rfcn_2d5_5x5.prototxt \
#  --weights data/imagenet_models/VGG16.v2.caffemodel \
#  --imdb kitti_train\
#  --iters 80000\
#  --cfg experiments/cfgs/kitti_rcnn_vgg16_multiscale_2d5.yml \

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/kitti_val/test_rcnn_multiscale_rfcn_2d5_nosub.prototxt \
  --net output/kitti/kitti_train/vgg16_fast_rcnn_multiscale_rfcn2d5_nosub_kitti_iter_40000.caffemodel \
  --imdb kitti_val \
  --cfg experiments/cfgs/kitti_rcnn_vgg16_multiscale_2d5_nosub.yml








