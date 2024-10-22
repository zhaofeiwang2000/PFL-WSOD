from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.boxes as box_utils
import utils.blob as blob_utils
import utils.net as net_utils
from core.config import cfg

import numpy as np
from sklearn.cluster import KMeans

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3



def PCL(boxes, cls_prob, im_labels, cls_prob_new):

    cls_prob = cls_prob.data.cpu().numpy()
    cls_prob_new = cls_prob_new.data.cpu().numpy()
    if cls_prob.shape[1] != im_labels.shape[1]:
        cls_prob = cls_prob[:, 1:]
    eps = 1e-9
    cls_prob[cls_prob < eps] = eps
    cls_prob[cls_prob > 1 - eps] = 1 - eps
    cls_prob_new[cls_prob_new < eps] = eps
    cls_prob_new[cls_prob_new > 1 - eps] = 1 - eps

    proposals = _get_graph_centers(boxes.copy(), cls_prob.copy(),
        im_labels.copy())

    labels, cls_loss_weights, gt_assignment, bbox_targets, bbox_inside_weights, bbox_outside_weights \
        = get_proposal_clusters(boxes.copy(), proposals, im_labels.copy())

    return {'labels' : labels.reshape(1, -1).astype(np.int64).copy(),
            'cls_loss_weights' : cls_loss_weights.reshape(1, -1).astype(np.float32).copy(),
            'gt_assignment' : gt_assignment.reshape(1, -1).astype(np.float32).copy(),
            'bbox_targets' : bbox_targets.astype(np.float32).copy(),
            'bbox_inside_weights' : bbox_inside_weights.astype(np.float32).copy(),
            'bbox_outside_weights' : bbox_outside_weights.astype(np.float32).copy()}

def NGIS(boxes, cls_prob, im_labels, cls_prob_new, N_Cosine):
    cls_prob = cls_prob.data.cpu().numpy()
    N_cos_score = N_Cosine.data.cpu().numpy()
    cls_prob_new = cls_prob_new.data.cpu().numpy()
    if cls_prob.shape[1] != im_labels.shape[1]:
        cls_prob = cls_prob[:, 1:]
    # import pdb; pdb.set_trace()
    eps = 1e-9
    cls_prob[cls_prob < eps] = eps
    cls_prob[cls_prob > 1 - eps] = 1 - eps
   
    proposals = _get_NGIS_proposals(boxes, cls_prob, im_labels, N_cos_score)
    proposals_out, max_index_all = _get_highest_score_proposals(boxes, cls_prob, im_labels)
    labels, cls_loss_weights, gt_assignment, bbox_targets, bbox_inside_weights, bbox_outside_weights \
        = get_proposal_clusters(boxes.copy(), proposals, im_labels.copy())
    return {'labels' : labels.reshape(1, -1).astype(np.int64).copy(),
            'cls_loss_weights' : cls_loss_weights.reshape(1, -1).astype(np.float32).copy(),
            'gt_assignment' : gt_assignment.reshape(1, -1).astype(np.float32).copy(),
            'bbox_targets' : bbox_targets.astype(np.float32).copy(),
            'bbox_inside_weights' : bbox_inside_weights.astype(np.float32).copy(),
            'bbox_outside_weights' : bbox_outside_weights.astype(np.float32).copy(),
            'highest_proposals': proposals_out,
            'max_index_all': max_index_all}

def _get_NGIS_proposals(boxes, cls_prob, im_labels, N_cos_score):
    """Get proposals with MIST."""
    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[0, :]
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)

    num_rois = boxes.shape[0]
    k = int(num_rois * 0.15)
    num_gt_cls = int(im_labels_tmp.sum())
    gt_cls_inds = im_labels_tmp[:].nonzero()[0]
    max_inds = np.argsort(-cls_prob[:, gt_cls_inds], axis=0)[:k]
    sorted_scores = cls_prob[max_inds.T]
    _boxes = boxes[max_inds.T]
    _N_cos_score = N_cos_score[max_inds.T]


    for num in range(_boxes.shape[0]):
        k_ind = np.zeros(k, dtype=np.int32)
        k_ind[0] = 1 # always take the one with max score 
        ious = box_utils.bbox_overlaps(_boxes[num].astype(dtype=np.float32, copy=False),_boxes[num].astype(dtype=np.float32, copy=False))
        for ii in range(1, k) :
            max_iou = np.max(ious[ii:ii+1, :ii], axis=1)
            k_ind[ii] = (max_iou < 0.2)*(_N_cos_score[num][ii]>3*_N_cos_score[num].mean())
        
        temp_cls = np.ones(k) * gt_cls_inds[num]
        classes = temp_cls[k_ind==1]
        scores = sorted_scores[num][k_ind==1][:, gt_cls_inds[num]]

        gt_boxes = np.vstack((gt_boxes, _boxes[num][k_ind==1]))
        gt_classes = np.vstack((gt_classes, 1+classes.reshape(-1,1)))
        gt_scores = np.vstack((gt_scores, scores.reshape(-1,1)))

    proposals = {'gt_boxes' : gt_boxes,
                 'gt_classes': gt_classes,
                 'gt_scores': gt_scores}

    return proposals


def MIST(boxes, cls_prob, im_labels, cls_prob_new, le_weight=0.5):
    #print(le_weight)
    cls_prob = cls_prob.data.cpu().numpy()
    cls_prob_new = cls_prob_new.data.cpu().numpy()
    if cls_prob.shape[1] != im_labels.shape[1]:
        cls_prob = cls_prob[:, 1:]
    eps = 1e-9
    cls_prob[cls_prob < eps] = eps
    cls_prob[cls_prob > 1 - eps] = 1 - eps
   
    proposals = _get_MIST_proposals(boxes, cls_prob, im_labels)
    proposals_out, max_index_all = _get_highest_score_proposals(boxes, cls_prob, im_labels)
    labels, cls_loss_weights, gt_assignment, bbox_targets, bbox_inside_weights, bbox_outside_weights \
        = get_proposal_clusters(boxes.copy(), proposals, im_labels.copy())
    return {'labels' : labels.reshape(1, -1).astype(np.int64).copy(),
            'cls_loss_weights' : cls_loss_weights.reshape(1, -1).astype(np.float32).copy(),
            'gt_assignment' : gt_assignment.reshape(1, -1).astype(np.float32).copy(),
            'bbox_targets' : bbox_targets.astype(np.float32).copy(),
            'bbox_inside_weights' : bbox_inside_weights.astype(np.float32).copy(),
            'bbox_outside_weights' : bbox_outside_weights.astype(np.float32).copy(),
            'highest_proposals': proposals_out,
            'max_index_all': max_index_all}


def _get_MIST_proposals(boxes, cls_prob, im_labels):
    """Get proposals with MIST."""
    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    #print(im_labels[0, :])
    im_labels_tmp = im_labels[0, :]
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)

    num_rois = boxes.shape[0]
    k = int(num_rois * 0.15)
    num_gt_cls = int(im_labels_tmp.sum())
    gt_cls_inds = im_labels_tmp[:].nonzero()[0]
    max_inds = np.argsort(-cls_prob[:, gt_cls_inds], axis=0)[:k]
    sorted_scores = cls_prob[max_inds.T]
    _boxes = boxes[max_inds.T]
    for num in range(_boxes.shape[0]):
        k_ind = np.zeros(k, dtype=np.int32)
        k_ind[0] = 1 # always take the one with max score 
        ious = box_utils.bbox_overlaps(_boxes[num].astype(dtype=np.float32, copy=False),_boxes[num].astype(dtype=np.float32, copy=False))
        for ii in range(1, k) :
            max_iou = np.max(ious[ii:ii+1, :ii], axis=1)
            k_ind[ii] = (max_iou < 0.2)
        temp_cls = np.ones(k) * gt_cls_inds[num]
        classes = temp_cls[k_ind==1]
        scores = sorted_scores[num][k_ind==1][:, gt_cls_inds[num]]

        gt_boxes = np.vstack((gt_boxes, _boxes[num][k_ind==1]))
        gt_classes = np.vstack((gt_classes, 1+classes.reshape(-1,1)))
        gt_scores = np.vstack((gt_scores, scores.reshape(-1,1)))

    proposals = {'gt_boxes' : gt_boxes,
                 'gt_classes': gt_classes,
                 'gt_scores': gt_scores}

    return proposals

def OICR(boxes, cls_prob, im_labels, cls_prob_new):

    cls_prob = cls_prob.data.cpu().numpy()
    cls_prob_new = cls_prob_new.data.cpu().numpy()
    if cls_prob.shape[1] != im_labels.shape[1]:
        cls_prob = cls_prob[:, 1:]
    eps = 1e-9
    cls_prob[cls_prob < eps] = eps
    cls_prob[cls_prob > 1 - eps] = 1 - eps

    proposals, max_index_all = _get_highest_score_proposals(boxes, cls_prob, im_labels)

    labels, cls_loss_weights, gt_assignment, bbox_targets, bbox_inside_weights, bbox_outside_weights \
        = get_proposal_clusters(boxes.copy(), proposals, im_labels.copy())

    return {'labels' : labels.reshape(1, -1).astype(np.int64).copy(),
            'cls_loss_weights' : cls_loss_weights.reshape(1, -1).astype(np.float32).copy(),
            'gt_assignment' : gt_assignment.reshape(1, -1).astype(np.float32).copy(),
            'bbox_targets' : bbox_targets.astype(np.float32).copy(),
            'bbox_inside_weights' : bbox_inside_weights.astype(np.float32).copy(),
            'bbox_outside_weights' : bbox_outside_weights.astype(np.float32).copy(),
            'highest_proposals': proposals,
            'max_index_all': max_index_all}


def _get_highest_score_proposals(boxes, cls_prob, im_labels):
    """Get proposals with highest score."""

    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[0, :]
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)
    max_index_all =[]
    for i in xrange(num_classes):
        if im_labels_tmp[i] == 1:
            cls_prob_tmp = cls_prob[:, i].copy()
            max_index = np.argmax(cls_prob_tmp)

            gt_boxes = np.vstack((gt_boxes, boxes[max_index, :].reshape(1, -1)))
            gt_classes = np.vstack((gt_classes, (i + 1) * np.ones((1, 1), dtype=np.int32)))
            gt_scores = np.vstack((gt_scores,
                                   cls_prob_tmp[max_index] * np.ones((1, 1), dtype=np.float32)))
            # cls_prob[max_index, :] = 0
            max_index_all.append(max_index)

    proposals = {'gt_boxes' : gt_boxes,
                 'gt_classes': gt_classes,
                 'gt_scores': gt_scores}

    return proposals, max_index_all


def _get_top_ranking_propoals(probs):
    """Get top ranking proposals by k-means"""
    kmeans = KMeans(n_clusters=cfg.TRAIN.NUM_KMEANS_CLUSTER,
        random_state=cfg.RNG_SEED).fit(probs)
    high_score_label = np.argmax(kmeans.cluster_centers_)

    index = np.where(kmeans.labels_ == high_score_label)[0]

    if len(index) == 0:
        index = np.array([np.argmax(probs)])

    return index


def _build_graph(boxes, iou_threshold):
    """Build graph based on box IoU"""
    overlaps = box_utils.bbox_overlaps(
        boxes.astype(dtype=np.float32, copy=False),
        boxes.astype(dtype=np.float32, copy=False))

    return (overlaps > iou_threshold).astype(np.float32)


def _get_graph_centers(boxes, cls_prob, im_labels):
    """Get graph centers."""

    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[0, :].copy()
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)
    for i in xrange(num_classes):
        if im_labels_tmp[i] == 1:
            cls_prob_tmp = cls_prob[:, i].copy()
            idxs = np.where(cls_prob_tmp >= 0)[0]
            idxs_tmp = _get_top_ranking_propoals(cls_prob_tmp[idxs].reshape(-1, 1))
            idxs = idxs[idxs_tmp]
            boxes_tmp = boxes[idxs, :].copy()
            cls_prob_tmp = cls_prob_tmp[idxs]

            graph = _build_graph(boxes_tmp, cfg.TRAIN.GRAPH_IOU_THRESHOLD)

            keep_idxs = []
            gt_scores_tmp = []
            count = cls_prob_tmp.size
            while True:
                order = np.sum(graph, axis=1).argsort()[::-1]
                tmp = order[0]
                keep_idxs.append(tmp)
                inds = np.where(graph[tmp, :] > 0)[0]
                gt_scores_tmp.append(np.max(cls_prob_tmp[inds]))

                graph[:, inds] = 0
                graph[inds, :] = 0
                count = count - len(inds)
                if count <= 5:
                    break

            gt_boxes_tmp = boxes_tmp[keep_idxs, :].copy()
            gt_scores_tmp = np.array(gt_scores_tmp).copy()

            keep_idxs_new = np.argsort(gt_scores_tmp)\
                [-1:(-1 - min(len(gt_scores_tmp), cfg.TRAIN.MAX_PC_NUM)):-1]

            gt_boxes = np.vstack((gt_boxes, gt_boxes_tmp[keep_idxs_new, :]))
            gt_scores = np.vstack((gt_scores,
                gt_scores_tmp[keep_idxs_new].reshape(-1, 1)))
            gt_classes = np.vstack((gt_classes,
                (i + 1) * np.ones((len(keep_idxs_new), 1), dtype=np.int32)))

            # If a proposal is chosen as a cluster center,
            # we simply delete a proposal from the candidata proposal pool,
            # because we found that the results of different strategies are similar and this strategy is more efficient
            cls_prob = np.delete(cls_prob.copy(), idxs[keep_idxs][keep_idxs_new], axis=0)
            boxes = np.delete(boxes.copy(), idxs[keep_idxs][keep_idxs_new], axis=0)

    proposals = {'gt_boxes' : gt_boxes,
                 'gt_classes': gt_classes,
                 'gt_scores': gt_scores}

    return proposals


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = box_utils.bbox_transform_inv(ex_rois, gt_rois,
                                           cfg.MODEL.BBOX_REG_WEIGHTS)
    return np.hstack((labels[:, np.newaxis], targets)).astype(
        np.float32, copy=False)


def _expand_bbox_targets(bbox_target_data):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.
    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.
    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    num_bbox_reg_classes = cfg.MODEL.NUM_CLASSES + 1

    clss = bbox_target_data[:, 0]
    bbox_targets = blob_utils.zeros((clss.size, 4 * num_bbox_reg_classes))
    bbox_inside_weights = blob_utils.zeros(bbox_targets.shape)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = (1.0, 1.0, 1.0, 1.0)
    return bbox_targets, bbox_inside_weights

def get_proposal_clusters(all_rois, proposals, im_labels):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    # overlaps: (rois x gt_boxes)
    gt_boxes = proposals['gt_boxes']
    gt_labels = proposals['gt_classes']
    gt_scores = proposals['gt_scores']
    overlaps = box_utils.bbox_overlaps(
        all_rois.astype(dtype=np.float32, copy=False),
        gt_boxes.astype(dtype=np.float32, copy=False))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_labels[gt_assignment, 0]
    cls_loss_weights = gt_scores[gt_assignment, 0]
   
    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]

    # Select background RoIs as those with < FG_THRESH overlap
    bg_inds = np.where(max_overlaps < cfg.TRAIN.FG_THRESH)[0]

    labels[bg_inds] = 0

    if cfg.MODEL.WITH_FRCNN:
        bbox_targets = _compute_targets(all_rois, gt_boxes[gt_assignment, :],
            labels)
        bbox_targets, bbox_inside_weights = _expand_bbox_targets(bbox_targets)
        bbox_outside_weights = np.array(
            bbox_inside_weights > 0, dtype=bbox_inside_weights.dtype) \
            * cls_loss_weights.reshape(-1, 1)
    else:
        bbox_targets, bbox_inside_weights, bbox_outside_weights = np.array([0]), np.array([0]), np.array([0])

    gt_assignment[bg_inds] = -1

    return labels, cls_loss_weights, gt_assignment, bbox_targets, bbox_inside_weights, bbox_outside_weights

class PCLLosses(nn.Module):

    def forward(ctx, pcl_probs, labels, cls_loss_weights, gt_assignments):
        cls_loss = 0.0
        weight = cls_loss_weights.view(-1).float()
        labels = labels.view(-1)
        gt_assignments = gt_assignments.view(-1)

        for gt_assignment in gt_assignments.unique():
            inds = torch.nonzero(gt_assignment == gt_assignments,
                as_tuple=False).view(-1)
            if gt_assignment == -1:
                assert labels[inds].sum() == 0
                cls_loss -= (torch.log(pcl_probs[inds, 0].clamp(1e-9, 10000))
                         * weight[inds]).sum()
            else:
                assert labels[inds].unique().size(0) == 1
                label_cur = labels[inds[0]]
                cls_loss -= torch.log(
                    pcl_probs[inds, label_cur].clamp(1e-9,  10000).mean()
                    ) * weight[inds].sum()

        return cls_loss / max(float(pcl_probs.size(0)), 1.)


class OICRLosses(nn.Module):
    def __init__(self):
        super(OICRLosses, self).__init__()

    def forward(self, prob, labels, cls_loss_weights, gt_assignments, eps = 1e-6):
        loss = torch.log(prob + eps)[range(prob.size(0)), labels]
        loss *= -cls_loss_weights
        ret = loss.mean()
        return ret
