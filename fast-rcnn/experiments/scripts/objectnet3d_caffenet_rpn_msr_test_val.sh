#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/objectnet3d_caffenet_rpn_msr_test_val.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/objectnet3d/test_rpn_msr.prototxt \
  --net output/objectnet3d/objectnet3d_trainval/caffenet_fast_rcnn_rpn_msr_objectnet3d_iter_160000.caffemodel \
  --imdb objectnet3d_val \
  --cfg experiments/cfgs/objectnet3d_rpn_msr.yml
