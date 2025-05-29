# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]


import torch
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .util import box_ops
import torch.nn.functional as F


def prepare_for_cdn(dn_args, training, num_queries, num_classes, hidden_dim, label_enc):
    """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
    if training:
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
        # positive and negative dn queries
        dn_number = dn_number * 2
        known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            dn_number = 1
        else:
            if dn_number >= 100:
                dn_number = dn_number // (int(max(known_num) * 2))
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t['labels'] for t in targets])    # 展平batch
        boxes = torch.cat([t['boxes'] for t in targets])
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])  # 记录batch id

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of bbox prob
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_pad = int(max(known_num))

        pad_size = int(single_pad * 2 * dn_number)
        positive_idx = torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part,
                                                  diff).cuda() * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expaned.long().to('cuda')
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True

        attn_mask[:pad_size, pad_size:] = True    # ours

        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True

        dn_meta = {
            'pad_size': pad_size,
            'num_dn_group': dn_number,
            'single_pad': single_pad,    # for ours
        }
    else:

        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None

    return input_query_label, input_query_bbox, attn_mask, dn_meta


def dn_post_process(outputs_class, outputs_coord, outputs_iou, binary_outputs_class, dn_meta, aux_loss, _set_aux_loss):
    """
        post process of dn after output from the transformer
        put the dn part in the dn_meta
    """
    if dn_meta and dn_meta['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :dn_meta['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :dn_meta['pad_size'], :]
        output_known_binary = binary_outputs_class[:, :, :dn_meta['pad_size'], :]

        outputs_class = outputs_class[:, :, dn_meta['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, dn_meta['pad_size']:, :]
        binary_outputs_class = binary_outputs_class[:, :, dn_meta['pad_size']:, :]

        if outputs_iou is not None:
            output_known_iou = outputs_iou[:, :, :dn_meta['pad_size'], :]
            outputs_iou = outputs_iou[:, :, dn_meta['pad_size']:, :]
            out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1], "pred_boxious": output_known_iou[-1],
                   'binary_outputs_class': output_known_binary[-1]}
        else:
            out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1], 'binary_outputs_class':output_known_binary[-1]}


        if aux_loss:
            if outputs_iou is not None:
                out['aux_outputs'] = _set_aux_loss(output_known_class, output_known_coord, output_known_iou, output_known_binary)
            else:
                out['aux_outputs'] = _set_aux_loss(output_known_class, output_known_coord, output_known_binary)
        dn_meta['output_known_lbs_bboxes'] = out
    if outputs_iou is not None:
        return outputs_class, outputs_coord, outputs_iou, binary_outputs_class
    else:
        return outputs_class, outputs_coord, binary_outputs_class


def dn_post_process_append(outputs_class, outputs_coord, outputs_binary, output_wh_list, dn_meta, aux_loss, _set_aux_loss):
    """
        post process of dn after output from the transformer
        put the dn part in the dn_meta
    """
    out_dn = []
    out_pred = []
    if dn_meta and dn_meta['pad_size'] > 0:
        for i in range(len(outputs_class)):
            class_ = outputs_class[i]
            coord_ = outputs_coord[i]
            binary_ = outputs_binary[i]
            dn_class = class_[:, :dn_meta['pad_size'], :]
            dn_coord = coord_[:, :dn_meta['pad_size'], :]
            dn_binary = binary_[:, :dn_meta['pad_size'], :]
            cls = class_[:, dn_meta['pad_size']:, :]
            coors = coord_[:, dn_meta['pad_size']:, :]
            binarys = binary_[:, dn_meta['pad_size']:, :]

            if len(output_wh_list) > 0:
                wh_ = output_wh_list[i]
                dn_wh = wh_[:, :dn_meta['pad_size'], :]
                wh_mat = wh_[:, dn_meta['pad_size']:, :]
            else:
                dn_wh = None
                wh_mat = None

            out = {'pred_logits':cls, 'pred_boxes':coors, 'pred_wh':wh_mat}
            out_dn_aux = {'pred_logits':dn_class, 'pred_boxes':dn_coord, 'pred_binary':dn_binary, 'pred_wh': dn_wh}
            
            out_pred.append(out)
            out_dn.append(out_dn_aux)

        out_dns = out_dn[-1]
        if aux_loss:
            out_dns['aux_outputs'] = out_dn[:-1]
        dn_meta['output_known_lbs_bboxes'] = out_dns

    return out_pred



def dn_post_process_append_zero_pad(outputs_class, outputs_coord, outputs_iou, outputs_binary, output_wh_list, dn_meta, aux_loss, _set_aux_loss):
    """
        post process of dn after output from the transformer
        put the dn part in the dn_meta
    """

    out_pred = []
    for i in range(len(outputs_class)):
        class_ = outputs_class[i]
        coord_ = outputs_coord[i]
        if outputs_iou is not None:
            iou_ = outputs_iou[i]
        if outputs_binary is not None:
            binary_ = outputs_binary[i]
        if len(output_wh_list) > 0:
            wh = output_wh_list[i]
        else:
            wh = None

        if outputs_binary is not None:
            if outputs_iou is None:
                out = {'pred_logits':class_, 'pred_boxes':coord_, 'pred_binary': binary_, 'pred_wh':wh}
            else:
                out = {'pred_logits':class_, 'pred_boxes':coord_, 'pred_boxious':iou_, 'pred_binary': binary_, 'pred_wh':wh}

        else:
            if outputs_iou is None:
                out = {'pred_logits':class_, 'pred_boxes':coord_, 'pred_wh':wh}
            else:
                out = {'pred_logits':class_, 'pred_boxes':coord_, 'pred_boxious':iou_, 'pred_wh':wh}
        out_pred.append(out)

    return out_pred