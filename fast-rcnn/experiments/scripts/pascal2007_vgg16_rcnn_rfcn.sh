#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/pascal2007_vgg16_rcnn_rfcn_1d0_sub6.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/VGG16/pascal2007/solver_rcnn_rfcn_7x7_sub6.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb voc_2007_trainval \
  --iters 90000 \
  --cfg experiments/cfgs/pascal_rcnn_rfcn_1d0.yml
  
time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/pascal2007/test_rcnn_rfcn_7x7_sub6.prototxt \
  --net output/pascal2007/voc_2007_trainval/vgg16_fast_rcnn_rfcn_7x7_sub6_pascal2007_iter_90000.caffemodel \
  --imdb voc_2007_test \
  --cfg experiments/cfgs/pascal_rcnn_rfcn_1d0.yml
  
#time ./tools/train_net.py --gpu $1 \
#  --solver models/VGG16/pascal2007/solver_rcnn_rfcn_7x7_bbox_nosub.prototxt \
#  --weights data/imagenet_models/VGG16.v2.caffemodel \
#  --imdb voc_2007_trainval \
#  --iters 80000 \
#  --cfg experiments/cfgs/pascal_rcnn_rfcn_1d0_nosub.yml
#  
#time ./tools/test_net.py --gpu $1 \
#  --def models/VGG16/pascal2007/test_rcnn_rfcn_7x7_1d0_bbox.prototxt \
#  --net output/pascal2007/voc_2007_trainval/vgg16_fast_rcnn_rfcn_7x7_1d0_bbox_nosub_pascal2007_iter_80000.caffemodel \
#  --imdb voc_2007_test \
#  --cfg experiments/cfgs/pascal_rcnn_rfcn_1d0_nosub.yml
#  
  
  
