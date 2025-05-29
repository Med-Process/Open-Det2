# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch
from torch import nn, Tensor

import copy

import math
import torch.nn.functional as F
from torch import nn

from .util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, compute_iou, box_iou


def gen_encoder_output_proposals(memory:Tensor, memory_padding_mask:Tensor, spatial_shapes:Tensor, learnedwh=None):
    """
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
        - learnedwh: 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    N_, S_, C_ = memory.shape
    base_scale = 4.0
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

        grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                        torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1) # H_, W_, 2

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

        if learnedwh is not None:
            wh = torch.ones_like(grid) * learnedwh.sigmoid() * (2.0 ** lvl)
        else:
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)

        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += (H_ * W_)

    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals)) # unsigmoid
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

    output_memory = memory
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    return output_memory, output_proposals


class RandomBoxPerturber():
    def __init__(self, x_noise_scale=0.2, y_noise_scale=0.2, w_noise_scale=0.2, h_noise_scale=0.2) -> None:
        self.noise_scale = torch.Tensor([x_noise_scale, y_noise_scale, w_noise_scale, h_noise_scale])

    def __call__(self, refanchors: Tensor) -> Tensor:
        nq, bs, query_dim = refanchors.shape
        device = refanchors.device

        noise_raw = torch.rand_like(refanchors)
        noise_scale = self.noise_scale.to(device)[:query_dim]

        new_refanchors = refanchors * (1 + (noise_raw - 0.5) * noise_scale)
        return new_refanchors.clamp_(0, 1)


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss


    # alpha = 1
    # if alpha >= 0:
    #     alpha_t = alpha * targets + (alpha) * (1 - targets)
    #     loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def sigmoid_loss_align_iou_score(inputs, targets, num_boxes, target_box, src_boxes, target_classes_o, align_score, alpha: float = 0.25, gamma: float = 2, idx=None):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()

    alpha = 0.25        
    iou = torch.diag(box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_box))[0])
    iou = torch.clamp(iou, 0.01)

    # text query alignment
    cls_align = align_score.sigmoid()

    pos_id_c = idx + (target_classes_o, )    # batch, id, class
    # t = prob[pos_id_c]**alpha * iou ** (1-alpha) * cls_align ** alpha

    t = prob[pos_id_c]**alpha * iou ** (1-2*alpha) * cls_align ** alpha
    t = torch.sqrt(t)    # 乘数因子

    t = torch.clamp(t, 0.1).detach()

    #compute classification loss
    #define hyper parameters here
    gamma = 2
        
    #define positive weights for SoftBceLoss        
    pos_weights=torch.zeros_like(prob, dtype=torch.float32)
    pos_weights[pos_id_c] =  (t -prob[pos_id_c]) ** gamma
    
    #define negative weights for SoftBceLoss
    neg_weights =  1 * prob ** gamma 
    neg_weights[pos_id_c] =  (1-t)* prob[pos_id_c] ** gamma

    loss = -pos_weights * prob.log() - neg_weights * (1-prob).log() 
    return loss.sum() / num_boxes



def normalize_iou(iou):
    if iou.shape[0] <= 0:
        return None
    iou_2 = iou ** 2
    max_ = iou.max()
    min_ = iou.min()

    max_2 = iou_2.max()
    scale = max_ / (max_2 + 1e-8)

    new_iou = iou_2 * scale
    return new_iou




class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MLP_BBox(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x, fea_return=False):
        final_fea = None
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

            if i == 1 and fea_return:
                final_fea = x
        if fea_return:
            return x, final_fea
        else:
            return x


def _get_activation_fn(activation, d_model=256, batch_dim=0):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu

    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def token_sigmoid_binary_focal_loss(pred_logits, targets, alpha=0.25, gamma=2.0, text_mask=None, reduction=True):
    # binary version of focal loss
    # copied from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # pred_logits: (bs, n_anchors, max_seq_len)
    # targets: (bs, n_anchors, max_seq_len)
    # text_mask: (bs, max_seq_len)
    # assert (targets.dim() == 3)
    # assert (pred_logits.dim() == 3)  # batch x from x to

    # bs, n, _ = pred_logits.shape
    if text_mask is not None:
        # assert (text_mask.dim() == 2)
        # text_mask = (text_mask > 0).unsqueeze(1) # (bs, 1, max_seq_len)
        # text_mask = text_mask.repeat(1, pred_logits.size(1), 1)  # copy along the image channel dimension. (bs, n_anchors, max_seq_len)
        pred_logits = torch.masked_select(pred_logits, text_mask)
        targets = torch.masked_select(targets, text_mask)

        # print(pred_logits.shape)
        # print(targets.shape)

    # targets[targets > 1.0] = 1.0

    p = torch.sigmoid(pred_logits)
    ce_loss = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction="none")

    # pred_logits = (pred_logits + 1.0) / 2.0
    # p = pred_logits
    # ce_loss = F.binary_cross_entropy(pred_logits, targets, reduction="none")
    
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction:
        return loss.sum() / targets.sum()

        # print((loss * targets).sum() / (targets.sum() + 0.0000001), (loss * (1-targets)).sum() / (1 - targets).sum())

        # if  (loss * targets).sum() / (targets.sum() + 0.0000001) + (loss * (1-targets)).sum() / (1 - targets).sum() > 1000:
        #     print( (loss * targets).sum() / (targets.sum() + 0.0000001) + (loss * (1-targets)).sum() / (1 - targets).sum())
        #     print()
        # return (loss * targets).sum() / (targets.sum() + 0.0000001) + (loss * (1-targets)).sum() / (1 - targets).sum()
    else:
        return loss
    

