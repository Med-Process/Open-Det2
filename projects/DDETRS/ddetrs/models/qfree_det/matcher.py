# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modules to compute the matching cost and solve the corresponding LSAP.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


import torch, os
from torch import nn
from scipy.optimize import linear_sum_assignment

from .util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, compute_iou, box_iou

import torch.nn.functional as F
import copy


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_opt: bool = False, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_binary: float = 1, focal_alpha = 0.25):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_binary = cost_binary

        self.cost_opt = cost_opt

        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs, targets, final_match=False):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = outputs["pred_logits"].shape[:2]


        # We flatten to compute the cost matrices in a batch
        out_prob = (outputs["pred_logits"]).flatten(0, 1).sigmoid() 

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        cost_class_binary = None
        final_match = True
        if final_match and 'binary_outputs_class' in outputs:
            binary_prob = outputs['binary_outputs_class'].flatten(0, 1).sigmoid()

            alpha = self.focal_alpha
            gamma = 2.0
            neg_cost_class_binary = (1 - alpha) * (binary_prob ** gamma) * (-(1 - binary_prob + 1e-8).log())
            pos_cost_class_binary = alpha * ((1 - binary_prob) ** gamma) * (-(binary_prob + 1e-8).log())
            cost_class_binary = pos_cost_class_binary - neg_cost_class_binary

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])


        # Compute the classification cost.
        if not self.cost_opt:
            alpha = self.focal_alpha
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            # cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            cost_class = torch.zeros((out_prob.size(0), tgt_ids.size(0)), device=out_prob.device)
            batch_size = 0
            for target in targets:
                num_label = len(target["labels"])
                srt = batch_size * num_queries
                end = (batch_size + 1) * num_queries
                cost_class[srt:end, :num_label] = pos_cost_class[srt:end, :num_label] - neg_cost_class[srt:end, :num_label]
                batch_size = batch_size + 1

        else:
            alpha = 0.25        
            iou = box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))[0]
            iou = torch.clamp(iou, 0.01)

            # batch之间需要隔离
            temp_mask = torch.zeros_like(iou)
            all_num = 0
            for i in range(bs):
                num_boxs = targets[i]['boxes'].shape[0]
                temp_mask[i*num_queries:(i+1)*num_queries, all_num:all_num+num_boxs] = 1
                all_num = all_num + num_boxs
            iou = iou * temp_mask

            t = out_prob[:, tgt_ids]**alpha * iou ** (1-alpha)

            t = torch.clamp(t, 0.1).detach()

            #compute classification loss
            #define hyper parameters here
            gamma = 2
                
            #define positive weights for SoftBceLoss        
            pos_weights=torch.zeros_like(out_prob, dtype=torch.float32)
            pos_weights =  (t -out_prob[:, tgt_ids]) ** gamma
            
            #define negative weights for SoftBceLoss
            neg_weights =  (1-t)* out_prob[:, tgt_ids] ** gamma

            cost_class = -pos_weights * out_prob.log()[:, tgt_ids] - neg_weights * (1-out_prob).log()[:, tgt_ids]


        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes            
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        # Final cost matrix
        if cost_class_binary is not None:
            N, M = cost_class.shape
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + cost_class_binary.repeat(1, M)
        else:
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou    # 原DINO

        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class HungarianMatcher_One_to_Many(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_opt: bool=False, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_binary: float = 1, focal_alpha = 0.25):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_binary = cost_binary

        self.cost_opt = cost_opt

        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

        self.focal_alpha = focal_alpha

        self.one_to_many = 2

    @torch.no_grad()
    def forward(self, outputs, targets, final_match=False):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = outputs["pred_logits"].shape[:2]

        new_targets = []
        for i in range(bs):
            tar = copy.deepcopy(targets[i]["boxes"].repeat(self.one_to_many, 1))
            labels = copy.deepcopy(targets[i]["labels"].repeat(self.one_to_many))
            new_targets.append({"boxes":tar, "labels":labels})


        out_prob = (outputs["pred_logits"]).flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in new_targets])
        tgt_bbox = torch.cat([v["boxes"] for v in new_targets])

        cost_class_binary = None
        final_match = True
        if final_match and 'binary_outputs_class' in outputs:
            binary_prob = outputs['binary_outputs_class'].flatten(0, 1).sigmoid()

            alpha = self.focal_alpha
            gamma = 2.0
            neg_cost_class_binary = (1 - alpha) * (binary_prob ** gamma) * (-(1 - binary_prob + 1e-8).log())
            pos_cost_class_binary = alpha * ((1 - binary_prob) ** gamma) * (-(binary_prob + 1e-8).log())
            cost_class_binary = pos_cost_class_binary - neg_cost_class_binary


        # Compute the classification cost.
        if not self.cost_opt:
            alpha = self.focal_alpha
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            
            cost_class = torch.zeros((out_prob.size(0), tgt_ids.size(0)), device=out_prob.device)
            batch_size = 0
            for target in targets:
                num_label = len(target["labels"])
                srt = batch_size * num_queries
                end = (batch_size + 1) * num_queries
                cost_class[srt:end, :num_label] = pos_cost_class[srt:end, :num_label] - neg_cost_class[srt:end, :num_label]
                batch_size = batch_size + 1
        else:
            alpha = 0.25        
            iou = box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))[0]
            iou = torch.clamp(iou, 0.01)

            temp_mask = torch.zeros_like(iou)
            all_num = 0
            for i in range(bs):
                num_boxs = targets[i]['boxes'].shape[0]
                temp_mask[i*num_queries:(i+1)*num_queries, all_num:all_num+num_boxs] = 1
                all_num = all_num + num_boxs
            iou = iou * temp_mask

            t = out_prob[:, tgt_ids]**alpha * iou ** (1-alpha)

            t = torch.clamp(t, 0.1).detach()

            #compute classification loss
            #define hyper parameters here
            gamma = 2
                
            #define positive weights for SoftBceLoss        
            pos_weights=torch.zeros_like(out_prob, dtype=torch.float32)
            pos_weights =  (t -out_prob[:, tgt_ids]) ** gamma
            
            #define negative weights for SoftBceLoss
            neg_weights =  (1-t)* out_prob[:, tgt_ids] ** gamma

            cost_class = -pos_weights * out_prob.log()[:, tgt_ids] - neg_weights * (1-out_prob).log()[:, tgt_ids]


        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes            
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        # Final cost matrix
        if cost_class_binary is not None:
            N, M = cost_class.shape
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou + cost_class_binary.repeat(1, M)
        else:
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou    # 原DINO

        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in new_targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], new_targets, self.one_to_many




class SimpleMinsumMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha = 0.25):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            
        # Compute the giou cost betwen boxes            
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1)

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        device = C.device
        for i, (c, _size) in enumerate(zip(C.split(sizes, -1), sizes)):
            weight_mat = c[i]
            idx_i = weight_mat.min(0)[1]
            idx_j = torch.arange(_size).to(device)
            indices.append((idx_i, idx_j))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    assert args.matcher_type in ['HungarianMatcher', 'SimpleMinsumMatcher'], "Unknown args.matcher_type: {}".format(args.matcher_type)
    if args.matcher_type == 'HungarianMatcher':
        return HungarianMatcher(cost_opt=args.match_cost_opt,
            cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou, cost_binary=args.set_cost_binary,
            focal_alpha=args.focal_alpha
        )
    elif args.matcher_type == 'SimpleMinsumMatcher':
        return SimpleMinsumMatcher(
            cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou, cost_binary=args.set_cost_binary,
            focal_alpha=args.focal_alpha
        )    
    else:
        raise NotImplementedError("Unknown args.matcher_type: {}".format(args.matcher_type))