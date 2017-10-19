#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/objectnet3d_vgg16_rpn_msr_test_2.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/objectnet3d/test_rpn_msr.prototxt \
  --net output/objectnet3d/objectnet3d_trainval/vgg16_fast_rcnn_rpn_msr_objectnet3d_iter_160000.caffemodel \
  --imdb objectnet3d_test_2 \
  --cfg experiments/cfgs/objectnet3d_rpn_msr.yml
