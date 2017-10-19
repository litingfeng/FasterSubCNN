# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from utils.cython_nms import nms, nms_new
from utils.boxes_grid import get_boxes_grid
import cPickle
import heapq
from utils.blob import im_list_to_blob
import os
import math
from rpn_msr.generate import imdb_proposals_det

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    processed_ims = []
    im_scale_factors = []
    scales = cfg.TEST.SCALES_BASE

    for im_scale in scales:
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)
    scales = np.array(scales)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    if cfg.IS_RPN:
        blobs = {'data' : None, 'boxes_grid' : None}
        blobs['data'], im_scale_factors = _get_image_blob(im)
        blobs['boxes_grid'] = rois
    else:
        blobs = {'data' : None, 'rois' : None}
        blobs['data'], im_scale_factors = _get_image_blob(im)
        if cfg.IS_MULTISCALE:
            if cfg.IS_EXTRAPOLATING:
                blobs['rois'] = _get_rois_blob(rois, cfg.TEST.SCALES)
            else:
                blobs['rois'] = _get_rois_blob(rois, cfg.TEST.SCALES_BASE)
        else:
            blobs['rois'] = _get_rois_blob(rois, cfg.TEST.SCALES_BASE)

    return blobs, im_scale_factors


def _get_patch_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""

    # process image
    im = im.astype(np.float32, copy=True)
    im -= cfg.PIXEL_MEANS
    height = im.shape[0]
    width = im.shape[1]

    num_rois = rois.shape[0]
    blob = np.zeros((num_rois, 224, 224, 3), dtype=np.float32)

    for i in xrange(num_rois):
        x1 = max(np.floor(rois[i, 0]), 1)
        y1 = max(np.floor(rois[i, 1]), 1)
        x2 = min(np.ceil(rois[i, 2]), width)
        y2 = min(np.ceil(rois[i, 3]), height)

        # crop image
        im_crop = im[y1:y2, x1:x2, :]

        # resize the cropped image
        blob[i, :, :, :] = cv2.resize(im_crop, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)

    return blob


def _bbox_pred(boxes, box_deltas):
    """Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + cfg.EPS
    heights = boxes[:, 3] - boxes[:, 1] + cfg.EPS
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = box_deltas[:, 0::4]
    dy = box_deltas[:, 1::4]
    dw = box_deltas[:, 2::4]
    dh = box_deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes


def _rescale_boxes(boxes, inds, scales):
    """Rescale boxes according to image rescaling."""

    for i in xrange(boxes.shape[0]):
        boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

    return boxes


def im_detect(net, im, boxes, num_classes, num_subclasses):
    """Detect object classes in an image given object proposals.
    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    if boxes.shape[0] == 0:
        scores = np.zeros((0, num_classes))
        pred_boxes = np.zeros((0, 4*num_classes))
        pred_views = np.zeros((0, 3*num_classes))
        scores_subcls = np.zeros((0, num_subclasses))
        return scores, pred_boxes, scores_subcls, pred_views

    blobs, unused_im_scale_factors = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['rois'].reshape(*(blobs['rois'].shape))
    blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False),
                            rois=blobs['rois'].astype(np.float32, copy=False))

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']

    if cfg.TEST.SUBCLS:
        scores_subcls = blobs_out['subcls_prob']
    else:
        # just use class scores
        scores_subcls = scores

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = _bbox_pred(boxes, box_deltas)
        pred_boxes = _clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.TEST.VIEWPOINT:
        # Apply bounding-box regression deltas
        pred_views = blobs_out['view_pred']
    else:
        # set to zeros
        pred_views = np.zeros((boxes.shape[0], 3*num_classes))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.IS_PATCH:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        scores_subcls = scores_subcls[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]
        pred_views = pred_views[inv_index, :]

    return scores, pred_boxes, scores_subcls, pred_views


def im_detect_patch(net, im, boxes, num_classes, num_subclasses):
    """Detect object classes in an image given object proposals.
    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    if boxes.shape[0] == 0:
        scores = np.zeros((0, num_classes))
        pred_boxes = np.zeros((0, 4*num_classes))
        pred_views = np.zeros((0, 3*num_classes))
        scores_subcls = np.zeros((0, num_subclasses))
        return scores, pred_boxes, scores_subcls, pred_views

    blob = _get_patch_blobs(im, boxes)

    # counting
    num = blob.shape[0]
    batchsize = 128
    num_batches = int(math.ceil(float(num) / float(batchsize)))

    # storage
    scores = np.zeros((num, num_classes))
    pred_boxes = np.zeros((num, 4*num_classes))
    pred_views = np.zeros((num, 3*num_classes))
    scores_subcls = np.zeros((num, num_subclasses))

    for batch_id in range(num_batches):
        start = batch_id * batchsize
        end = (batch_id+1) * batchsize
        if end > num:
            end = num

        blobs = {'data' : None}
        blobs['data'] = blob[start:end, :, :, :]

        # reshape network inputs
        net.blobs['data'].reshape(*(blobs['data'].shape))
        blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False))

        if cfg.TEST.SVM:
            # use the raw scores before softmax under the assumption they
            # were trained as linear SVMs
            scores[start:end, :] = net.blobs['cls_score'].data
        else:
            # use softmax estimated probabilities
            scores[start:end, :] = blobs_out['cls_prob']

        if cfg.TEST.SUBCLS:
            scores_subcls[start:end, :] = blobs_out['subcls_prob']
        else:
            # just use class scores
            scores_subcls[start:end, :] = scores[start:end, :]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = blobs_out['bbox_pred']
            pred_boxes[start:end, :] = _bbox_pred(boxes[start:end, :], box_deltas)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes[start:end, :] = np.tile(boxes[start:end, :], (1, scores.shape[1]))

        if cfg.TEST.VIEWPOINT:
            # Apply bounding-box regression deltas
            pred_views[start:end, :] = blobs_out['view_pred']

    if cfg.TEST.BBOX_REG:
        pred_boxes = _clip_boxes(pred_boxes, im.shape)

    return scores, pred_boxes, scores_subcls, pred_views



def im_detect_proposal(net, im, boxes_grid, num_classes, num_subclasses, subclass_mapping):
    """Detect object classes in an image given boxes on grids.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of boxes

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    blobs, im_scale_factors = _get_blobs(im, boxes_grid)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['boxes_grid'].reshape(*(blobs['boxes_grid'].shape))
    blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False),
                            boxes_grid=blobs['boxes_grid'].astype(np.float32, copy=False))

    scores_subcls = blobs_out['subcls_prob']
    print scores_subcls.shape

    # build max_scores
    tmp = np.reshape(scores_subcls, (scores_subcls.shape[0], scores_subcls.shape[1]))
    if cfg.TEST.SUBCLS:
        max_scores = np.zeros((scores_subcls.shape[0], num_classes))
        max_scores[:,0] = tmp[:,0]
        assert (num_classes == 2 or num_classes == 4 or num_classes == 13 or num_classes == 21), 'The number of classes is not supported!'
        if num_classes == 2:
            max_scores[:,1] = tmp[:,1:].max(axis = 1)
        else:
            for i in xrange(1, num_classes):
                index = np.where(subclass_mapping == i)[0]
                max_scores[:,i] = tmp[:,index].max(axis = 1)
        scores = max_scores
    else:
        scores = tmp

    rois = net.blobs['rois_sub'].data
    inds = rois[:,0]
    boxes = rois[:,1:]
    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = _bbox_pred(boxes, box_deltas)
        pred_boxes = _rescale_boxes(pred_boxes, inds, cfg.TRAIN.SCALES)
        pred_boxes = _clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))
        pred_boxes = _rescale_boxes(pred_boxes, inds, cfg.TRAIN.SCALES)
        pred_boxes = _clip_boxes(pred_boxes, im.shape)

    if cfg.TEST.VIEWPOINT:
        # Apply bounding-box regression deltas
        pred_views = blobs_out['view_pred']
    else:
        # set to zeros
        pred_views = np.zeros((boxes.shape[0], 3*num_classes))

    # only select one aspect with the highest score
    """
    num = boxes.shape[0]
    num_aspect = len(cfg.TEST.ASPECTS)
    inds = []
    for i in xrange(num/num_aspect):
        index = range(i*num_aspect, (i+1)*num_aspect)
        max_scores = scores[index,1:].max(axis = 1)
        ind_max = np.argmax(max_scores)
        inds.append(index[ind_max])
    """

    # select boxes
    max_scores = scores[:,1:].max(axis = 1)
    labels = scores[:,1:].argmax(axis = 1) + 1
    order = max_scores.ravel().argsort()[::-1]
    # inds = np.where(max_scores > cfg.TEST.ROI_THRESHOLD)[0]
    inds = order[:cfg.TEST.ROI_NUM]
    scores = scores[inds]
    pred_boxes = pred_boxes[inds]
    pred_views = pred_views[inds]
    scores_subcls = scores_subcls[inds]
    labels = labels[inds]
    print scores.shape
   
    # draw boxes
    if 0:
        # print scores, pred_boxes.shape
        import matplotlib.pyplot as plt
        plt.imshow(im)
        for j in xrange(pred_boxes.shape[0]):
            roi = pred_boxes[j,4:]
            plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                           roi[3] - roi[1], fill=False,
                           edgecolor='g', linewidth=3))
        plt.show()

    # conv5 = net.blobs['conv5'].data

    return scores, pred_boxes, scores_subcls, labels, pred_views

def vis_detections(im, class_name, dets, thresh=0.1):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    plt.cla()
    plt.imshow(im)
    for i in xrange(np.minimum(1, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, 4]
        view = dets[i, 6:9]
        if score > thresh:
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3))
            plt.title('{}  {:.3f}, view {:.1f} {:.1f} {:.1f}'.format(class_name, score, \
                      view[0] * 180 / math.pi, view[1] * 180 / math.pi, view[2] * 180 / math.pi))
    plt.show()

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue

            x1 = dets[:, 0]
            y1 = dets[:, 1]
            x2 = dets[:, 2]
            y2 = dets[:, 3]
            scores = dets[:, 4]
            inds = np.where((x2 > x1) & (y2 > y1) & (scores > cfg.TEST.DET_THRESHOLD))[0]
            dets = dets[inds,:]
            if dets == []:
                continue

            keep = nms(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def test_net(net, imdb):

    output_dir = get_output_dir(imdb, net)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if 'nissan' in imdb.name:
        output_dir_center = os.path.join(output_dir, 'imagesCenter')
        if not os.path.exists(output_dir_center):
            os.makedirs(output_dir_center)
        output_dir_left = os.path.join(output_dir, 'imagesLeft')
        if not os.path.exists(output_dir_left):
            os.makedirs(output_dir_left)
        output_dir_right = os.path.join(output_dir, 'imagesRight')
        if not os.path.exists(output_dir_right):
            os.makedirs(output_dir_right)

    det_file = os.path.join(output_dir, 'detections.pkl')
    print imdb.name
    # if os.path.exists(det_file):
    #     with open(det_file, 'rb') as fid:
    #         all_boxes = cPickle.load(fid)
    #     print 'Detections loaded from {}'.format(det_file)

    #     if cfg.IS_RPN:
    #         print 'Evaluating detections'
    #         imdb.evaluate_proposals(all_boxes, output_dir)
    #     else:
    #         print 'Applying NMS to all detections'
    #         nms_dets = apply_nms(all_boxes, cfg.TEST.NMS)
    #         print 'Evaluating detections'
    #         print imdb.name
    #         if not 'objectnet3d' in imdb.name:
    #             imdb.evaluate_detections(nms_dets, output_dir)
    #         #imdb.evaluate_detections_one_file(nms_dets, output_dir)
    #     return

    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # heuristic: keep an average of 40 detections per class per images prior
    # to NMS
    if ('voc' in imdb.name or 'pascal' in imdb.name or 'objectnet3d' in imdb.name) and cfg.IS_RPN == False:
        max_per_set = 40 * num_images
        max_per_image = 100
    else:
        max_per_set = np.inf
        # heuristic: keep at most 100 detection per class per image prior to NMS
        max_per_image = 10000
        # detection thresold for each class (this is adaptively set based on the
        # max_per_set constraint)

    if cfg.IS_RPN:
        thresh = -np.inf * np.ones(imdb.num_classes)
    else:
        thresh = cfg.TEST.DET_THRESHOLD * np.ones(imdb.num_classes)
        # top_scores will hold one minheap of scores per class (used to enforce the max_per_set constraint)
        top_scores = [[] for _ in xrange(imdb.num_classes)]

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    if cfg.IS_RPN == False:
        roidb = imdb.roidb

    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i))

        _t['im_detect'].tic()
        if cfg.IS_RPN:
            boxes_grid, _, _ = get_boxes_grid(im.shape[0], im.shape[1])
            scores, boxes, scores_subcls, labels, views = im_detect_proposal(net, im, boxes_grid, imdb.num_classes, imdb.num_subclasses, imdb.subclass_mapping)

            # save conv5 features
            # index = imdb._image_index[i]
            # filename = os.path.join(output_dir, index[5:] + '_conv5.pkl')
            # with open(filename, 'wb') as f:
            #    cPickle.dump(conv5, f, cPickle.HIGHEST_PROTOCOL)
        else:
            if cfg.TEST.IS_PATCH:
                scores, boxes, scores_subcls, views = im_detect_patch(net, im, roidb[i]['boxes'], imdb.num_classes, imdb.num_subclasses)
            else:
                scores, boxes, scores_subcls, views = im_detect(net, im, roidb[i]['boxes'], imdb.num_classes, imdb.num_subclasses)
        #print "scores_subcls ", scores_subcls.shape
        scores_subcls = scores_subcls[:,:,0,0]
        _t['im_detect'].toc()

        _t['misc'].tic()
        count = 0
        for j in xrange(1, imdb.num_classes):
            if cfg.IS_RPN:
                # inds = np.where(scores[:, j] > thresh[j])[0]
                inds = np.where(labels == j)[0]
            else:
                inds = np.where((scores[:, j] > thresh[j]) & (roidb[i]['gt_classes'] == 0))[0]

            cls_scores = scores[inds, j]
            subcls_scores = scores_subcls[inds, :]
            #print "subcls_scores ", subcls_scores.shape
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_views = views[inds, j*3:(j+1)*3]

            top_inds = np.argsort(-cls_scores)[:max_per_image]
            cls_scores = cls_scores[top_inds]
            subcls_scores = subcls_scores[top_inds, :]
            #print "subcls_scores ", subcls_scores.shape
            cls_boxes = cls_boxes[top_inds, :]
            cls_views = cls_views[top_inds, :]

            if cfg.IS_RPN == False:
                # push new scores onto the minheap
                for val in cls_scores:
                    heapq.heappush(top_scores[j], val)
                # if we've collected more than the max number of detection,
                # then pop items off the minheap and update the class threshold
                if len(top_scores[j]) > max_per_set:
                    while len(top_scores[j]) > max_per_set:
                        heapq.heappop(top_scores[j])
                    thresh[j] = top_scores[j][0]

            # select the maximum score subclass in this class
            if cfg.TEST.SUBCLS and cfg.IS_RPN == False:
                index = np.where(imdb.subclass_mapping == j)[0]
                #print "index ", index.shape
                max_indexes = subcls_scores[:,index].argmax(axis = 1)
                #print "max_indexes ", max_indexes.shape
                sub_classes = index[max_indexes]
                #print "subclasses ", sub_classes.shape
            else:
                if subcls_scores.shape[0] == 0:
                    sub_classes = cls_scores
                else:
                    sub_classes = subcls_scores.argmax(axis = 1).ravel()

            #print "cls_boxes ", cls_boxes.shape
            #print "cls_scores ", cls_scores.shape
            #print "sub_classes ", sub_classes.shape
            #print "cls_views ", cls_views.shape
            all_boxes[j][i] = \
                    np.hstack((cls_boxes, cls_scores[:, np.newaxis], sub_classes[:, np.newaxis], cls_views)) \
                    .astype(np.float32, copy=False)
            count = count + len(cls_scores)

            if 0:
                keep = nms(all_boxes[j][i], cfg.TEST.NMS)
                vis_detections(im, imdb.classes[j], all_boxes[j][i][keep, :])
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:d} object detected {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, count, _t['im_detect'].average_time, _t['misc'].average_time)
        #assert(1==0)
    for j in xrange(1, imdb.num_classes):
        for i in xrange(num_images):
            inds = np.where(all_boxes[j][i][:, 4] > thresh[j])[0]
            all_boxes[j][i] = all_boxes[j][i][inds, :]

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    if cfg.IS_RPN:
        print 'Evaluating detections'
        imdb.evaluate_proposals(all_boxes, output_dir)
        if 'mot' in imdb.name:
            imdb.evaluate_proposals_one_file(all_boxes, output_dir)
    else:
        print 'Applying NMS to all detections'
        nms_dets = apply_nms(all_boxes, cfg.TEST.NMS)
        print 'Evaluating detections'
        if not 'objectnet3d' in imdb.name:
            imdb.evaluate_detections(nms_dets, output_dir)
        #imdb.evaluate_detections_one_file(nms_dets, output_dir)


def test_rpn_msr_net(net, imdb):

    output_dir = get_output_dir(imdb, net)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    det_file = os.path.join(output_dir, 'detections.pkl')
    if os.path.exists(det_file):
        with open(det_file, 'rb') as fid:
            all_boxes = cPickle.load(fid)
        print 'Detections loaded from {}'.format(det_file)

        print 'Evaluating detections'
        imdb.evaluate_proposals_msr(all_boxes, output_dir)
        return

    # Generate proposals on the imdb
    all_boxes = imdb_proposals_det(net, imdb)

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb.evaluate_proposals_msr(all_boxes, output_dir)
