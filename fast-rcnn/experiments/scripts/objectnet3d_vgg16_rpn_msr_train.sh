#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/objectnet3d_vgg16_rpn_msr_train.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/VGG16/objectnet3d/solver_rpn_msr.prototxt \
  --weights data/imagenet_models/VGG16.v2.caffemodel \
  --imdb objectnet3d_trainval \
  --cfg experiments/cfgs/objectnet3d_rpn_msr.yml \
  --iters 160000
