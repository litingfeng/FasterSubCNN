<<<<<<< HEAD
# SubCNN: Subcategory-aware Convolutional Neural Networks for Object Proposals and Detection

Created by Yu Xiang at CVGL at Stanford University.

Part of this code is based on the [Fast R-CNN](https://github.com/rbgirshick/fast-rcnn) created by Ross Girshick at Microsoft Research, Redmond.

### Introduction

We introduce a new region proposal network that uses subcategory information to guide the proposal generating process, and a new detection network for joint detection and subcategory classification. By using subcategories related to object pose, we achieve state-of-the-art performance on both detection and pose estimation on commonly used benchmarks, such as [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object_detail.php?result=5e17cbbabbf775d8cc376793168be49bd6f01608), [PASCAL3D+](http://cvgl.stanford.edu/projects/pascal3d.html) and [ObjectNet3D](http://cvgl.stanford.edu/projects/objectnet3d/).

This package supports
 - Subcategory-aware region proposal network
 - Subcategory-aware detection network
 - Region proposal network in Faster R-CNN (Ren et al. NIPS 2015)
 - Detection network in Faster R-CNN (Ren et al. NIPS 2015)
 - Experiments on the following datasets: KITTI Detection, PASCAL VOC, PASCAL3D+, ObjectNet3D, KITTI Tracking sequences, MOT sequences

### License

SubCNN is released under the MIT License (refer to the LICENSE file for details).

### Citing SubCNN

If you find SubCNN useful in your research, please consider citing:

    @incollection{xiang2016subcategory,
        author = {Xiang, Yu and Choi, Wongun and Lin, Yuanqing and Savarese, Silvio},
        title = {Subcategory-aware Convolutional Neural Networks for Object Proposals and Detection},
        booktitle = {arXiv:1604.04693},
        year = {2016}
    }

### Installation

1. Clone the SubCNN repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/yuxng/SubCNN.git
  ```
  
2. We'll call the directory that you cloned SubCNN into `ROOT`

   *Ignore notes 1 and 2 if you followed step 1 above.*
   
   **Note 1:** If you didn't clone SubCNN with the `--recursive` flag, then you'll need to manually clone the `caffe-fast-rcnn` submodule:
    ```Shell
    git submodule update --init --recursive
    ```
    **Note 2:** The `caffe-fast-rcnn` submodule needs to be on the `fast-rcnn` branch (or equivalent detached state). This will happen automatically *if you follow these instructions*.

3. Build the Cython modules
    ```Shell
    cd $ROOT/fast-rcnn/lib
    make
    ```
    
4. Build our modified Caffe and pycaffe. Make sure you have cuDNN to save GPU memory.
    ```Shell
    cd $ROOT/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # In the Makefile.config, use CUSTOM_CXX := g++ -std=c++11

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```
    
5. Download pre-trained ImageNet models
    ```Shell
    cd $ROOT/fast-rcnn
    ./data/scripts/fetch_imagenet_models.sh
    ```

    This will populate the `$ROOT/fast-rcnn/data` folder with `imagenet_models`.

### Running with the KITTI detection dataset
1. Download the KITTI detection dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php).

2. Create symlinks for the KITTI detection dataset
    ```Shell
    cd $ROOT/fast-rcnn/data/KITTI
    ln -s $data_object_image_2 data_object_image_2
    ```

3. Unzip the voxel_exemplars.zip in $ROOT/fast-rcnn/data/KITTI. These are subcategories from [3D voxel patterns](http://cvgl.stanford.edu/projects/3DVP/) (Xiang et al. CVPR'15).

4. Run the region proposal network to generate region proposals
    ```Shell
    cd $ROOT/fast-rcnn

    # subcategory-aware RPN for validation
    ./experiments/scripts/kitti_val_caffenet_rpn.sh $GPU_ID

    # subcategory-aware RPN for testing
    ./experiments/scripts/kitti_test_caffenet_rpn_6k8k.sh $GPU_ID

    # Faster RCNN RPN for validation
    ./experiments/scripts/kitti_val_caffenet_rpn_msr.sh $GPU_ID

    # Faster RCNN RPN for testing
    ./experiments/scripts/kitti_test_caffenet_rpn_msr_6k8k.sh $GPU_ID

    ```

5. Copy the region proposals to $ROOT/fast-rcnn/data/KITTI/region_proposals/RPN_*:
    ```Shell
    # validation (125 subcategories for car)
    $ROOT/fast-rcnn/data/KITTI/region_proposals/RPN_125/training   # a directory contains region proposals for training images: 000000.txt, ..., 007480.txt

    # testing (227 subcategories for car)
    $ROOT/fast-rcnn/data/KITTI/region_proposals/RPN_227/training   # a directory contains region proposals for training images: 000000.txt, ..., 007480.txt
    $ROOT/fast-rcnn/data/KITTI/region_proposals/RPN_227/testing    # a directory contains region proposals for testing  images: 000000.txt, ..., 007517.txt
    ```

5. Run the detection network
    ```Shell
    cd $ROOT/fast-rcnn

    # subcategory-aware detection network for validation
    ./experiments/scripts/kitti_val_caffenet_rcnn_multiscale.sh $GPU_ID

    # subcategory-aware detection network for testing
    ./experiments/scripts/kitti_test_caffenet_rcnn_multiscale_6k8k.sh $GPU_ID

    # subcategory-aware detection network for testing with VGG16
    ./experiments/scripts/kitti_test_vgg16_rcnn_multiscale_6k8k.sh $GPU_ID

    # subcategory-aware detection network for testing with GoogleNet
    ./experiments/scripts/kitti_test_googlenet_rcnn.sh $GPU_ID

    # Faster RCNN detection network for validation
    ./experiments/scripts/kitti_val_caffenet_rcnn_msr.sh $GPU_ID

    # Faster RCNN detection network for testing
    ./experiments/scripts/kitti_test_caffenet_rcnn_original_msr.sh $GPU_ID

    ```

### Running with the PASCAL3D+ dataset
1. Download the PASCAL3D+ dataset from [here](http://cvgl.stanford.edu/projects/pascal3d.html).

2. Create symlinks for the PASCAL VOC 2012 dataset
    ```Shell
    cd $ROOT/fast-rcnn/data/PASCAL3D
    ln -s $PASCAL3D+_release1.1/PASCAL/VOCdevkit VOCdevkit2012
    ```

3. Unzip the voxel_exemplars.zip in $ROOT/fast-rcnn/data/PASCAL3D. These are subcategories from [3D voxel patterns](http://cvgl.stanford.edu/projects/3DVP/) (Xiang et al. CVPR'15).

4. Run the region proposal network to generate region proposals
    ```Shell
    cd $ROOT/fast-rcnn

    # subcategory-aware RPN
    ./experiments/scripts/pascal3d_vgg16_rpn_6k8k.sh $GPU_ID

    ```

5. Copy the region proposals to $ROOT/fast-rcnn/data/PASCAL3D/region_proposals/RPN_6k8k:
    ```Shell
    # training set and validation set
    $ROOT/fast-rcnn/data/PASCAL3D/region_proposals/RPN_6k8k/training     # a directory contains region proposals for training images
    $ROOT/fast-rcnn/data/PASCAL3D/region_proposals/RPN_6k8k/validation   # a directory contains region proposals for validation images

    ```

5. Run the detection network
    ```Shell
    cd $ROOT/fast-rcnn

    # subcategory-aware detection network
    ./experiments/scripts/pascal3d_vgg16_rcnn_multiscale.sh $GPU_ID

    ```

### Running with the ObjectNet3D dataset
1. Download the ObjectNet3D dataset from [here](http://cvgl.stanford.edu/projects/objectnet3d/).

2. Create symlinks for the ObjectNet3D dataset
    ```Shell
    cd $ROOT/fast-rcnn/data/ObjectNet3D
    ln -s $ObjectNet3D/Images Images
    ln -s $ObjectNet3D/Image_sets Image_sets
    ```

3. Write ObjectNet3D annotations to text files
    ```Shell
    cd $ROOT/ObjectNet3D
    # change the path of ObjectNet3D in globals.m
    write_annotations.m
    cp -r Labels $ROOT/fast-rcnn/data/ObjectNet3D/Labels
    ```

4. Run the region proposal network or scripts in $ROOT/ObjectNet3D to generate region proposals. You may need to download code for selective search, EdgeBoxes or MCG from the web.
    ```Shell
    # change the path of ObjectNet3D in globals.m
    # selective search
    cd $ROOT/ObjectNet3D
    selective_search_ObjectNet3D.m

    # edgeboxes
    cd $ROOT/ObjectNet3D
    edgeboxes_ObjectNet3D.m

    # mcg
    cd $ROOT/ObjectNet3D
    mcg_ObjectNet3D.m

    # Faster RCNN RPN
    cd $ROOT/fast-rcnn
    ./experiments/scripts/objectnet3d_vgg16_rpn_msr_train.sh $GPU_ID

    ```

5. Copy the region proposals to $ROOT/fast-rcnn/data/ObjectNet3D/region_proposals:
    ```Shell
    # a directory contains region proposals for selective search
    $ROOT/fast-rcnn/data/ObjectNet3D/region_proposals/selective_search

    # a directory contains region proposals for EdgeBoxes
    $ROOT/fast-rcnn/data/ObjectNet3D/region_proposals/edge_boxes

    # a directory contains region proposals for MCG
    $ROOT/fast-rcnn/data/ObjectNet3D/region_proposals/mcg

    # a directory contains region proposals for Faster RCNN RPN
    $ROOT/fast-rcnn/data/ObjectNet3D/region_proposals/rpn_vgg16

    ```

6. Run the detection and viewpoint estimation network
    ```Shell
    cd $ROOT/fast-rcnn

    # detection and viewpoint estimation with selective search region proposals
    ./experiments/scripts/objectnet3d_vgg16_rcnn_view_selective_search.sh $GPU_ID

    # detection and viewpoint estimation with EdgeBoxes region proposals
    ./experiments/scripts/objectnet3d_vgg16_rcnn_view_edge_boxes.sh $GPU_ID

    # detection and viewpoint estimation with MCG region proposals
    ./experiments/scripts/objectnet3d_vgg16_rcnn_view_mcg.sh $GPU_ID

    # detection and viewpoint estimation with RPN region proposals
    ./experiments/scripts/objectnet3d_vgg16_rcnn_view_rpn.sh $GPU_ID
    ```

### Running with other datasets
The package also supports running experiments on the PASCAL VOC detection dataset, the [KITTI Tracking dataset](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) and the [MOT Tracking dataset](https://motchallenge.net/data/2D_MOT_2015/). Please see the scripts in $ROOT/fast-rcnn/experiments/scripts.

### Our trained Models
You can download our trained models on the KITTI dataset, the PASCAL3D+ dataset and the ObjectNet3D dataset (2.2G) from ftp://cs.stanford.edu/cs/cvgl/SubCNN_models.zip

Please check the script [test_subcnn_models.sh](https://github.com/yuxng/SubCNN/blob/master/fast-rcnn/experiments/scripts/test_subcnn_models.sh) in SubCNN/fast-rcnn/experiments/scripts for usage of these trained models.


### Running with the NTHU dataset (internal usage)
1. The NTHU dataset should have a directory named 'data', under which it has the following structure:
    ```Shell
  	$data/                           # the directory contains all the data
  	$data/71                         # a directory for video 71: 000001.jpg, ..., 002956.jpg
  	$data/71.txt                     # a txt file contains the frame names: 000001 \n 000002 \n ... 002956
  	# ... and several other directories and txt files ...
    ```

2. Create symlinks for the NTHU dataset
    ```Shell
    cd $ROOT/fast-rcnn/data/NTHU
    ln -s $data data
    ```

3. Run the region proposal network to generate region proposals, modify the script to run with different videos
    ```Shell
    cd $ROOT/fast-rcnn
    ./experiments/scripts/nthu_caffenet_rpn_6k8k.sh $GPU_ID
    ```

4. Copy the region proposals to $ROOT/fast-rcnn/data/NTHU/region_proposals/RPN_6k8k:
    ```Shell
    $ROOT/fast-rcnn/data/NTHU/region_proposals/RPN_6k8k/71    # a directory contains region proposals for video 71: 000001.txt, ..., 002956.txt
    ```

5. Run the detection network, modify the script to run with different videos
    ```Shell
    cd $ROOT/fast-rcnn
    ./experiments/scripts/nthu_caffenet_rcnn_multiscale_6k8k.sh $GPU_ID
    ```
=======
# FasterSubCNN
Cost Efficient Subcategory-aware CNN for Object Detection
>>>>>>> 58aa5e3307218bd3569a09586d18e99d31f79cab
