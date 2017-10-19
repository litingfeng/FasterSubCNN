#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/test_subcnn_models.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

#######################
# ObjectNet3D test
#######################

time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/objectnet3d/test_rcnn_view.prototxt \
  --net data/SubCNN_models/objectnet3d_trainval/vgg16_fast_rcnn_view_objectnet3d_selective_search_iter_160000.caffemodel \
  --imdb objectnet3d_test \
  --cfg experiments/cfgs/objectnet3d_rcnn_view_selective_search.yml

##################
# KITTI valiation
##################

# detection on KITTI validation set with GoogleNet
time ./tools/test_net.py --gpu $1 \
  --def models/GoogleNet/kitti_val/test_rcnn.prototxt \
  --net data/SubCNN_models/kitti_train/googlenet_fast_rcnn_kitti_iter_40000.caffemodel \
  --imdb kitti_val \
  --cfg experiments/cfgs/kitti_rcnn_googlenet.yml

# detection on KITTI validation set with CaffeNet
time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/kitti_val/test_rcnn_multiscale.prototxt \
  --net data/SubCNN_models/kitti_train/caffenet_fast_rcnn_multiscale_kitti_iter_40000.caffemodel \
  --imdb kitti_val \
  --cfg experiments/cfgs/kitti_rcnn_multiscale.yml

# RPN on KITTI validation set
time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/kitti_val/test_rpn.prototxt \
  --net data/SubCNN_models/kitti_train/caffenet_fast_rcnn_rpn_kitti_iter_40000.caffemodel \
  --imdb kitti_val \
  --cfg experiments/cfgs/kitti_rpn.yml

##################
# KITTI test
##################

# detection on KITTI test set with GoogleNet
time ./tools/test_net.py --gpu $1 \
  --def models/GoogleNet/kitti_test/test_rcnn.prototxt \
  --net data/SubCNN_models/kitti_trainval/googlenet_fast_rcnn_kitti_iter_80000.caffemodel \
  --imdb kitti_test \
  --cfg experiments/cfgs/kitti_rcnn_googlenet.yml

# detection on KITTI test set with VGG16Net
time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/kitti_test/test_rcnn_multiscale.prototxt \
  --net data/SubCNN_models/kitti_trainval/vgg16_fast_rcnn_multiscale_6k8k_kitti_iter_80000.caffemodel \
  --imdb kitti_test \
  --cfg experiments/cfgs/kitti_rcnn_vgg16_multiscale_6k8k.yml

# detection on KITTI test set with CaffeNet
time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/kitti_test/test_rcnn_multiscale.prototxt \
  --net data/SubCNN_models/kitti_trainval/caffenet_fast_rcnn_multiscale_6k8k_kitti_iter_80000.caffemodel \
  --imdb kitti_test \
  --cfg experiments/cfgs/kitti_rcnn_multiscale_6k8k.yml

# RPN on KITTI test set
time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/kitti_test/test_rpn.prototxt \
  --net data/SubCNN_models/kitti_trainval/caffenet_fast_rcnn_rpn_6k8k_kitti_iter_80000.caffemodel \
  --imdb kitti_test \
  --cfg experiments/cfgs/kitti_rpn.yml

#######################
# PASCAL3D+ validation
#######################

# detection on PASCAL3D+ validation set with VGG16Net
time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/pascal3d/test_rcnn_multiscale.prototxt \
  --net data/SubCNN_models/pascal3d_train/vgg16_fast_rcnn_multiscale_pascal3d_iter_40000.caffemodel \
  --imdb pascal3d_val \
  --cfg experiments/cfgs/pascal3d_rcnn_multiscale.yml

# RPN on PASCAL3D+ validation set with VGG16Net
time ./tools/test_net.py --gpu $1 \
  --def models/VGG16/pascal3d/test_rpn.prototxt \
  --net data/SubCNN_models/pascal3d_train/vgg16_fast_rcnn_rpn_pascal3d_iter_40000.caffemodel \
  --imdb pascal3d_val \
  --cfg experiments/cfgs/pascal3d_rpn_vgg16.yml
