# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
import math
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from util.box_iou_my import bbox_iou_my

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss)

# from .deformable_transformer import build_deformable_transformer
from .deformable_transformer_small import build_deformable_transformer

from .deformable_decoder_transformer import build_deformable_decoder_transformer
from .utils import sigmoid_focal_loss, sigmoid_focal_loss_reweight, MLP

from ..registry import MODULE_BUILD_FUNCS
from .dn_components import prepare_for_cdn,dn_post_process, dn_post_process_append, dn_post_process_append_zero_pad

from .utils import sigmoid_focal_loss, sigmoid_focal_loss_me, sigmoid_focal_loss_reweight,  sigmoid_focal_loss_PSL, sigmoid_focal_loss_small, MLP


class DINO(nn.Module):
    """ This is the Cross-Attention Detector module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, 
                    aux_loss=False, iter_update=False,
                    query_dim=2, 
                    random_refpoints_xy=False,
                    fix_refpoints_hw=-1,
                    num_feature_levels=1,
                    nheads=8,
                    # two stage
                    two_stage_type='no', # ['no', 'standard']
                    two_stage_add_query_num=0,
                    dec_pred_class_embed_share=True,
                    dec_pred_bbox_embed_share=True,
                    two_stage_class_embed_share=True,
                    two_stage_bbox_embed_share=True,
                    decoder_sa_type = 'sa',
                    num_patterns = 0,
                    dn_number = 100,
                    dn_box_noise_scale = 0.4,
                    dn_label_noise_ratio = 0.5,
                    dn_labelbook_size = 100,
                    use_gqs_transformer=False,
                    nms_decoder=0,
                    binary=False,
                    wh_part=False,
                    ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.

            fix_refpoints_hw: -1(default): learn w and h for each box seperately
                                >0 : given fixed number
                                -2 : learn a shared w and h
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)
        self.binary = binary
        self.wh_part = wh_part

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4
        self.random_refpoints_xy = random_refpoints_xy
        self.fix_refpoints_hw = fix_refpoints_hw

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # for gcn
        self.use_gqs_transformer = use_gqs_transformer

        self.nms_decoder = nms_decoder

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == 'no', "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = nn.Linear(hidden_dim, num_classes)
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        # 用于预测bias
        # self.box_bias_my = MLP(hidden_dim, hidden_dim//2, hidden_dim, 2)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)]
        if dec_pred_class_embed_share:
            class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        else:
            class_embed_layerlist = [copy.deepcopy(_class_embed) for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        
        self.transformer.decoder.bbox_embed1 = self.bbox_embed
        self.transformer.decoder.class_embed1 = self.class_embed
        # self.transformer.decoder.mlp = MLP(hidden_dim, hidden_dim*8, hidden_dim, 2)


        _binary_embed = MLP(hidden_dim, hidden_dim, 1, 3)
        nn.init.constant_(_binary_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_binary_embed.layers[-1].bias.data, 0)
        binary_embed_layerlist = [copy.deepcopy(_binary_embed) for i in range(transformer.num_decoder_layers)]
        self.binary_embed = nn.ModuleList(binary_embed_layerlist)
        self.enc_binary = copy.deepcopy(_binary_embed)
        self.transformer.enc_binary_class = self.enc_binary
        self.transformer.decoder.binary_embed = self.binary_embed

        # embedding condition
        self.conditional_binary = nn.Embedding(91, hidden_dim)
        # self.conditional_binary_enc = nn.Embedding(91, hidden_dim)
        nn.init.constant_(self.conditional_binary.weight.data, 0)
        # nn.init.constant_(self.conditional_binary_enc.weight.data, 0)
        # self.conditional_binary = None
        self.conditional_binary_enc = None
        self.transformer.decoder.conditional_binary = self.conditional_binary
        self.transformer.conditional_binary_enc = self.conditional_binary_enc

        # 初始化单独预测wh分支
        if self.wh_part:
            part_wh_embed = MLP(hidden_dim, hidden_dim, 2, 3)
            nn.init.constant_(part_wh_embed.layers[-1].weight.data, 0)
            nn.init.constant_(part_wh_embed.layers[-1].bias.data, 0)
            part_embed_layerlist = [part_wh_embed for i in range(transformer.num_decoder_layers)]
            self.wh_part_embed = nn.ModuleList(part_embed_layerlist)

        # binary_class = nn.Linear(hidden_dim, 1)
        # binary_class.bias.data = torch.ones(1) * bias_value
        # self.transformer.enc_binary_class = copy.deepcopy(binary_class)
        # binary_layer_list = [copy.deepcopy(binary_class) for i in range(transformer.num_decoder_layers)]
        # self.binary_class = nn.ModuleList(binary_layer_list)
        # self.transformer.decoder.binary_class = self.binary_class
        # self.MLP = MLP(hidden_dim, hidden_dim*8, hidden_dim, 2)    # 为tgt做条件处理
        # self.transformer.enc_mlp = self.MLP


        # self.bbox_embed1 = nn.ModuleList(box_embed_layerlist)
        # self.class_embed1 = nn.ModuleList(class_embed_layerlist)
        # self.bbox_embed2 = nn.ModuleList(copy.deepcopy(box_embed_layerlist))
        # self.class_embed2 = nn.ModuleList(copy.deepcopy(class_embed_layerlist))

        # self.transformer.decoder.bbox_embed1 = self.bbox_embed1
        # self.transformer.decoder.class_embed1 = self.class_embed1
        # self.transformer.decoder.bbox_embed2 = self.bbox_embed2
        # self.transformer.decoder.class_embed2 = self.class_embed2

        # two stage
        self.two_stage_type = two_stage_type
        self.two_stage_add_query_num = two_stage_add_query_num
        assert two_stage_type in ['no', 'standard'], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type != 'no':
            if two_stage_bbox_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
                self.transformer.backbone_out_bbox_embed = copy.deepcopy(_bbox_embed)

                # self.transformer.enc_out_bbox_embed = [copy.deepcopy(_bbox_embed) for i in range(2)]
                # self.transformer.backbone_out_bbox_embed = [copy.deepcopy(_bbox_embed) for i in range(2)]

                # if use_gcn_transformer:
                #     self.transformer.enc_out_bbox_gcn_embed = copy.deepcopy(_bbox_embed)
    
            if two_stage_class_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)    # 原方法
                self.transformer.backbone_out_class_embed = copy.deepcopy(_class_embed)    # 用于backbone的分类

                # self.transformer.enc_out_class_embed = [copy.deepcopy(_class_embed) for i in range(2)]    # 原方法
                # self.transformer.backbone_out_class_embed = [copy.deepcopy(_class_embed) for i in range(2)]    # 用于backbone的分类

                # if use_gcn_transformer:
                #     self.transformer.enc_out_class_gcn_embed = copy.deepcopy(_class_embed)

    
            self.refpoint_embed = None
            if self.two_stage_add_query_num > 0:
                self.init_ref_points(two_stage_add_query_num)

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']
        if decoder_sa_type == 'ca_label':
            self.label_embedding = nn.Embedding(num_classes, hidden_dim)
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = self.label_embedding
        else:
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = None
            self.label_embedding = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)
        if self.random_refpoints_xy:

            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

        if self.fix_refpoints_hw > 0:
            print("fix_refpoints_hw: {}".format(self.fix_refpoints_hw))
            assert self.random_refpoints_xy
            self.refpoint_embed.weight.data[:, 2:] = self.fix_refpoints_hw
            self.refpoint_embed.weight.data[:, 2:] = inverse_sigmoid(self.refpoint_embed.weight.data[:, 2:])
            self.refpoint_embed.weight.data[:, 2:].requires_grad = False
        elif int(self.fix_refpoints_hw) == -1:
            pass
        elif int(self.fix_refpoints_hw) == -2:
            print('learn a shared h and w')
            assert self.random_refpoints_xy
            self.refpoint_embed = nn.Embedding(use_num_queries, 2)
            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False
            self.hw_embed = nn.Embedding(1, 1)
        else:
            raise NotImplementedError('Unknown fix_refpoints_hw {}'.format(self.fix_refpoints_hw))

    def save_visual(self, samples, targets):
        from util.visualizer import COCOVisualizer
        
        vslzr = COCOVisualizer()
        vslzr.visualize(samples.tensors[0].cpu(), targets[0], savedir="./out_fig", dpi=100)
        print("save data!...")



    def get_colors(self, pred, target):
        colors = []
        import random
        import numpy as np
        random.seed(1024)
        id = 0
        boxs = pred[id]["boxes"]
        xyxy = box_ops.box_cxcywh_to_xyxy(boxs) 
        H, W = target[id]['size'].tolist() 
        colors = []
        for mm in xyxy:
            ori_xyxy = mm.cpu().numpy() * np.array([W, H, W, H])
            size = int(ori_xyxy[2] - ori_xyxy[0]) * int(ori_xyxy[3] - ori_xyxy[1])
            # if size >= 32*32:
            #     r = random.randint(8,10) * 0.1
            #     g = random.randint(2,5) * 0.1
            #     b = random.randint(3,5) * 0.1
            #     # c = (np.random.random((1, 3))*0.3+0.7).tolist()[0]
            # else:
            #     # c = (np.random.random((1, 3))*0.7+0.1).tolist()[0]
            #     r = random.randint(2,5) * 0.1
            #     g = random.randint(2,5) * 0.1
            #     b = random.randint(8,10) * 0.1

            c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]

            colors.append([int(c[0]*255), int(c[1]*255), int(c[2]*255)])
        
        return colors

    def draw_pred_boxes(self, targets, img, out, topk_proposals, level_start_index):
        n, num = topk_proposals.shape

        level0 = level_start_index[0].item()
        level1 = level_start_index[1].item()
        select = []
        for i in range(num):
            id = topk_proposals[0, i].item()
            if id >= level0 and id < level1:  # 筛选第一层的query结果
                select.append(i)
        number_res = len(select)
        out_filter = copy.deepcopy(out)
        out_filter["pred_logits"] = out["pred_logits"][:, select,:]
        out_filter["pred_boxes"] = out["pred_boxes"][:, select,:]

        postprocessors = {'bbox': PostProcess(num_select=number_res, nms_iou_threshold=-1)}
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](out_filter, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        import cv2

        h, w, c = img.shape

        font = cv2.FONT_HERSHEY_SIMPLEX
        
        colors = self.get_colors(results, targets)

        for i in range(len(results[0]["scores"])):
            s = results[0]["scores"][i]
            if s >= 0.1:    # 0.5
                box = results[0]["boxes"][i]
                img = cv2.rectangle(img.copy(), (int(box[0]/orig_target_sizes[0][1]*w), int(box[1]/orig_target_sizes[0][0]*h)), 
                                    (int(box[2]/orig_target_sizes[0][1]*w), int(box[3]/orig_target_sizes[0][0]*h)), colors[i], 2)

                # cv2.putText(img, str(round(s.item(), 2)), (int(box[0]/orig_target_sizes[0][1]*w), int(box[1]/orig_target_sizes[0][0]*h - 5)), font, 0.6, colors[i], 2)

        cv2.putText(img, "Q-Num:" + str(topk_proposals.shape[1]), (int(20), int(20)), font, 0.6, (255, 255, 0), 2)

        return img


    def save_visual_query(self, level_start_index, spatial_shapes, topk_proposals, samples, targets, out):
        level0 = level_start_index[0].item()
        level1 = level_start_index[1].item()
        level2 = level_start_index[2].item()
        level3 = level_start_index[3].item()

        l1_h, l1_w = spatial_shapes[0][0], spatial_shapes[0][1]
        l2_h, l2_w = spatial_shapes[1][0], spatial_shapes[1][1]
        l3_h, l3_w = spatial_shapes[2][0], spatial_shapes[2][1]
        l4_h, l4_w = spatial_shapes[3][0], spatial_shapes[3][1]

        from util.visualizer import renorm
        import random
        import numpy as np
        batch_id = 0
        img = renorm(samples.tensors.cpu()[batch_id]).permute(1, 2, 0)
        img = np.uint8(img*255)
        img_ori = img.copy()    # 原图像
        h_, w_, c_ = img_ori.shape

        img_ori = self.draw_pred_boxes(targets, img_ori, out, topk_proposals, level_start_index)

        lv1 = torch.zeros((l1_h, l1_w, 3), dtype=torch.long)
        lv2 = torch.zeros((l2_h, l2_w), dtype=torch.long)
        lv3 = torch.zeros((l3_h, l3_w), dtype=torch.long)
        lv4 = torch.zeros((l4_h, l4_w), dtype=torch.long)


        n, num = topk_proposals.shape

        l0 = 0; l1 = 0; l2 = 0; l3 = 0
        for i in range(num):
            id = topk_proposals[0, i].item()
            if id >= level0 and id < level1:                
                hi = id // l1_w    # 行
                wj = id % l1_w     # 列
                lv1[hi.item(), wj.item(), 0] = int(255)
                hid = int(hi.item() / l1_h * h_)
                wid = int(wj.item() / l1_w * w_)
                # img_ori[hid, wid, 0] = 255
                # img_ori[hid, wid, 1] = 0
                # img_ori[hid, wid, 2] = 0
                l0 = l0 + 1
            elif id >= level1 and id < level2:
                id = id - level1
                hi = id // l2_w    # 行
                wj = id % l2_w     # 列
                lv2[hi.item(), wj.item()] = int(255)
                l1 = l1 + 1
            elif id >= level2 and id < level3:
                id = id - level2
                hi = id // l3_w    # 行
                wj = id % l3_w     # 列
                lv3[hi.item(), wj.item()] = int(255)
                l2 = l2 + 1
            elif id >= level3:
                id = id - level3
                hi = id // l4_w    # 行
                wj = id % l4_w     # 列
                lv4[hi.item(), wj.item()] = int(255)
                l3 = l3 + 1
        print("number: ", l0, l1, l2, l3)

        import cv2
        # lv1 = cv2.applyColorMap(cv2.convertScaleAbs(lv1.cpu().numpy(), alpha=1), cv2.COLORMAP_JET)
        # lv2 = cv2.applyColorMap(cv2.convertScaleAbs(lv2.cpu().numpy(), alpha=1), cv2.COLORMAP_JET)
        # lv3 = cv2.applyColorMap(cv2.convertScaleAbs(lv3.cpu().numpy(), alpha=1), cv2.COLORMAP_JET)
        # lv4 = cv2.applyColorMap(cv2.convertScaleAbs(lv4.cpu().numpy(), alpha=1), cv2.COLORMAP_JET)



        lv1 = np.uint8(lv1)
        lv1_resi = cv2.resize(lv1, (w_, h_), interpolation=cv2.INTER_LINEAR)
        result1 = cv2.addWeighted(img_ori, 1.0, lv1_resi, 1.0, 0)


        import time, random
        from datetime import datetime
        cur_time = time.time()
        cur_time = datetime.fromtimestamp(cur_time)    # 2023, 10, 28, 13, 24, 47, 10398
        cur_time = str(cur_time)
        save_name1 = "./out_fig/" + cur_time + "_" + str(targets[0]["image_id"].item()) + "level1.png" 
        save_name2 = "./out_fig/" + cur_time + "_" + str(targets[0]["image_id"].item()) + "level1_ori.png" 

        cv2.imwrite(save_name1, cv2.cvtColor(result1, cv2.COLOR_RGB2BGR))
        cv2.imwrite(save_name2, cv2.cvtColor(img_ori, cv2.COLOR_RGB2BGR))


    def pred_in_4_layers(self, layer1, layer2, layer_id, fea, level_start_index, topk_proposals):
        level0 = level_start_index[0].item()
        level1 = level_start_index[1].item()
        level2 = level_start_index[2].item()
        level3 = level_start_index[3].item()

        dn_num = fea.shape[1] - topk_proposals.shape[1]
        if dn_num > 0:
            fea_dn = fea[:, :dn_num, :]
            fea_q = fea[:, dn_num:, :]
        else:
            fea_q = fea

        id_sort_value, sort_indice = torch.sort(topk_proposals, 1)
        sort_fea = torch.gather(fea_q, 1, sort_indice.unsqueeze(dim=2).repeat(1, 1, 
                                                    fea_q.shape[2]))    # fea: 1 900 256
        v_, s_id = torch.sort(sort_indice, 1)

        # for layers
        idx0 = torch.where(id_sort_value > level1)[1][0]
        fea1 = layer1[layer_id](sort_fea[:, 0:idx0, :])
        # fea1 = sort_fea[:, 0:idx0, :]
        
        # idx1 = torch.where(id_sort_value > level2)[1][0]
        fea2 = layer2[layer_id](sort_fea[:, idx0:, :])
        # fea2 = sort_fea[:, idx0:, :]

        
        # idx2 = torch.where(id_sort_value > level3)[1][0]
        # fea3 = layer[2].cuda()[layer_id](sort_fea[:, idx1:idx2, :])
        
        # fea4 = layer[3].cuda()[layer_id](sort_fea[:, idx2:, :])

        new_fea = torch.cat([fea1, fea2], dim=1)
        new_fea = torch.gather(new_fea, 1, s_id.unsqueeze(dim=2).repeat(1, 1, 
                                                    new_fea.shape[2]))

        if dn_num > 0:
            new_fea = torch.cat([0.5 * (layer1[layer_id](fea_dn) + layer2[layer_id](fea_dn)), new_fea], dim=1)
            

        return new_fea

    def get_batch_box_number(self, targets):
        out = 0
        for i in range(len(targets)):
            t = targets[i]['boxes'].shape[0]
            if t > out:
                out = t
        
        return out



    def forward(self, samples: NestedTensor, targets:List=None, is_train=True):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        # self.save_visual(samples, targets)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        if self.dn_number > 0 or targets is not None:
            input_query_label, input_query_bbox, attn_mask, dn_meta =\
                prepare_for_cdn(dn_args=(targets, self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale),
                                training=self.training,num_queries=self.num_queries,num_classes=self.num_classes,
                                hidden_dim=self.hidden_dim,label_enc=self.label_enc)
        else:
            assert targets is None
            input_query_bbox = input_query_label = attn_mask = dn_meta = None

        if is_train:
            # max_boxes = self.get_batch_box_number(targets)
            max_boxes = 0
        else:
            max_boxes = 0
        hs, reference, nms_selected, dn_numbers, hs_enc, gcs_transformer_out, gcs_back_out, gqs_enc, ref_gqs_enc, ref_enc, init_box_proposal, tgt_undetach_bias_4, src_flatten_return, src_flatten_coord, level_start_index, spatial_shapes, memory, topk_proposals = self.transformer(srcs, masks, input_query_bbox, poss,input_query_label,attn_mask, max_boxes)

        # In case num object=0
        hs[0] += self.label_enc.weight[0,0]*0.0

        # deformable-detr-like anchor update
        # reference_before_sigmoid = inverse_sigmoid(reference[:-1]) # n_dec, bs, nq, 4
        outputs_coord_list = []
        output_wh_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs, nms_sd, dns, binary_layer) in enumerate(zip(reference[:-1], self.bbox_embed, hs, nms_selected, dn_numbers, self.binary_embed)):
            # binary_ = (binary_lyer(layer_hs).sigmoid() * 100).int()
            # b, n, _ = binary_.shape
            # weight_ = self.conditional_binary(binary_.view(-1))
            # layer_delta_unsig = layer_bbox_embed(layer_hs + weight_.view(b,n,-1))
            
            layer_delta_unsig = layer_bbox_embed(layer_hs)

            
            if not self.nms_decoder:
                layer_outputs_unsig = layer_delta_unsig  + inverse_sigmoid(layer_ref_sig)
            else:
                if dns > 0:
                    dn_part_ref_sig = layer_ref_sig[:, :dns, :]
                    ref_sig = layer_ref_sig[:, dns:, :]
                    ref_sig = torch.gather(ref_sig, 1, nms_sd.unsqueeze(-1).repeat(1, 1, 4))
                    ref_sig = torch.cat([dn_part_ref_sig, ref_sig], dim=1)
                else:
                    if layer_ref_sig.shape[1] > layer_delta_unsig.shape[1]:
                        ref_sig = torch.gather(layer_ref_sig, 1, nms_sd.unsqueeze(-1).repeat(1, 1, 4))
                    else:
                        ref_sig = layer_ref_sig                        
                layer_outputs_unsig = layer_delta_unsig  + inverse_sigmoid(ref_sig)
    
            if self.wh_part:
                layer_wh_part = self.wh_part_embed[dec_lid](layer_hs)
                wh = layer_wh_part + inverse_sigmoid(layer_ref_sig)[:, :, 2:]
                output_wh_list.append(wh.sigmoid())

            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        if not self.nms_decoder:
            outputs_coord_list = torch.stack(outputs_coord_list)
            outputs_class = torch.stack([layer_cls_embed(layer_hs) for
                                     layer_cls_embed, layer_hs in zip(self.class_embed, hs)])
            if self.wh_part:
                output_wh_list = torch.stack(output_wh_list)        
        else:
            outputs_class = [layer_cls_embed(layer_hs) for layer_cls_embed, layer_hs in zip(self.class_embed, hs)]

        if self.binary:
            # outputs_binary = torch.stack([layer_cls_embed(layer_hs.detach()) for
            #                          layer_cls_embed, layer_hs in zip(self.binary_embed, hs)])
            outputs_binary = [layer_cls_embed(layer_hs.detach()) for
                                     layer_cls_embed, layer_hs in zip(self.binary_embed, hs)]
        else:
            outputs_binary = None


        # 四层分别预测
        # for dec_lid, (layer_ref_sig, layer_hs) in enumerate(zip(reference[:-1], hs)):
        #     # layer_delta_unsig = layer_bbox_embed(layer_hs)
        #     layer_delta_unsig = self.pred_in_4_layers(self.bbox_embed1, self.bbox_embed2, dec_lid, layer_hs, level_start_index, topk_proposals)
        #     layer_outputs_unsig = layer_delta_unsig  + inverse_sigmoid(layer_ref_sig)
        #     layer_outputs_unsig = layer_outputs_unsig.sigmoid()
        #     outputs_coord_list.append(layer_outputs_unsig)
        # outputs_coord_list = torch.stack(outputs_coord_list)        

        # outputs_class = torch.stack([self.pred_in_4_layers(self.class_embed1, self.class_embed2, i, hs[i], level_start_index, topk_proposals) for i in range(len(hs))])


        # outputs_class = torch.stack([layer_cls_embed(self.class_em_my(layer_hs)) for
        #                              layer_cls_embed, layer_hs in zip(self.class_embed, hs)])


        if self.dn_number > 0 and dn_meta is not None:
            if not self.nms_decoder and not self.binary:
                outputs_class, outputs_coord_list = \
                    dn_post_process(outputs_class, outputs_coord_list,
                                    dn_meta,self.aux_loss,self._set_aux_loss)
                out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord_list[-1]}
            else:
                if dn_meta['pad_size'] > 0:
                    out_pred = dn_post_process_append(outputs_class, outputs_coord_list, outputs_binary, output_wh_list,
                                    dn_meta,self.aux_loss,self._set_aux_loss)
                    # outputs_class = outputs_class[:, :, dn_meta['pad_size']:, :]
                    # outputs_coord_list = outputs_coord_list[:, :, dn_meta['pad_size']:, :]
                    outputs_class = [outputs_class[i][:, dn_meta['pad_size']:, :] for i in range(len(outputs_class))]
                    outputs_coord_list = [outputs_coord_list[i][:, dn_meta['pad_size']:, :] for i in range(len(outputs_coord_list))]
                else:
                    out_pred = dn_post_process_append_zero_pad(outputs_class, outputs_coord_list, outputs_binary, output_wh_list,
                                    dn_meta,self.aux_loss,self._set_aux_loss)
                    # outputs_class = outputs_class[:, :, dn_meta['pad_size']:, :]
                    # outputs_coord_list = outputs_coord_list[:, :, dn_meta['pad_size']:, :]
                    outputs_class = [outputs_class[i][:, dn_meta['pad_size']:, :] for i in range(len(outputs_class))]
                    outputs_coord_list = [outputs_coord_list[i][:, dn_meta['pad_size']:, :] for i in range(len(outputs_coord_list))]
                out = out_pred[-1]
        else:
            out_pred = dn_post_process_append_zero_pad(outputs_class, outputs_coord_list, outputs_binary, output_wh_list,
                            dn_meta,self.aux_loss,self._set_aux_loss)
            out = out_pred[-1]

        if self.binary:
            if self.dn_number > 0 and dn_meta is not None:
                # out['pred_binary'] = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord_list[-1], 'pred_binary': outputs_binary[:, :, dn_meta['pad_size']:, :][-1]}
                # out['pred_binary'] = outputs_binary[:, :, dn_meta['pad_size']:, :][-1]
                out['pred_binary'] = outputs_binary[5][:, dn_meta['pad_size']:, :]
            else:
                # out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord_list[-1], 'pred_binary': outputs_binary[-1]}
                out['pred_binary'] = outputs_binary[-1]


        if self.aux_loss:
            if not self.nms_decoder:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)
            else:
                out['aux_outputs'] = out_pred[:-1]
            if self.binary:
                if self.dn_number > 0 and dn_meta is not None:
                    for i in range(len(out['aux_outputs'])):
                        # out['aux_outputs'][i]['pred_binary'] = outputs_binary[:, :, dn_meta['pad_size']:, :][:-1][i]
                        out['aux_outputs'][i]['pred_binary'] = outputs_binary[i][:, dn_meta['pad_size']:, :]
                    # out['aux_outputs'] = [{'pred_binary': a} for a in outputs_binary[:, :, dn_meta['pad_size']:, :][:-1]]
                else:
                    # out['aux_outputs'] = [{'pred_binary': a} for a in outputs_binary[:-1]]
                    for i in range(len(out['aux_outputs'])):
                        # out['aux_outputs'][i]['pred_binary'] = outputs_binary[:-1][i]
                        out['aux_outputs'][i]['pred_binary'] = outputs_binary[i]

            if self.wh_part:
                if self.dn_number > 0 and dn_meta is not None:
                    for i in range(len(out['aux_outputs'])):
                        out['aux_outputs'][i]['pred_wh'] = output_wh_list[:, :, dn_meta['pad_size']:, :][:-1][i]
                else:
                    for i in range(len(out['aux_outputs'])):
                        out['aux_outputs'][i]['pred_wh'] = output_wh_list[:-1][i]

        # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])
            if self.use_gqs_transformer:
                # interm_gqs_coord = ref_gqs_enc[-1]
                # tgt_undetach_bias = self.transformer.activation_gelu(self.transformer.bias_norm(gqs_enc[-1].detach()))
                # coords = self.transformer.tgt_bias_box(self.transformer.head_tgt_bias(tgt_undetach_bias))
                # ref = copy.deepcopy(reference[0][:, -self.num_queries:, :])
                # coord = (tgt_undetach_bias_4 + inverse_sigmoid(ref[:, :, :])).sigmoid()
                # interm_gqs_coord = coord
                # interm_gqs_class = self.transformer.enc_out_class_embed(gqs_enc[-1])
                # out['interm_outputs_gqs'] = {'pred_logits': interm_gqs_class, 'pred_boxes': interm_gqs_coord}
                pass

            out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
            # out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}
            out['interm_outputs']['pred_binary'] = self.transformer.enc_binary_class(hs_enc[-1].detach())
            

            # prepare enc outputs
            if hs_enc.shape[0] > 1:
                enc_outputs_coord = []
                enc_outputs_class = []
                for layer_id, (layer_box_embed, layer_class_embed, layer_hs_enc, layer_ref_enc) in enumerate(zip(self.enc_bbox_embed, self.enc_class_embed, hs_enc[:-1], ref_enc[:-1])):
                    layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
                    layer_enc_outputs_coord_unsig = layer_enc_delta_unsig + inverse_sigmoid(layer_ref_enc)
                    layer_enc_outputs_coord = layer_enc_outputs_coord_unsig.sigmoid()

                    layer_enc_outputs_class = layer_class_embed(layer_hs_enc)
                    enc_outputs_coord.append(layer_enc_outputs_coord)
                    enc_outputs_class.append(layer_enc_outputs_class)

                out['enc_outputs'] = [
                    {'pred_logits': a, 'pred_boxes': b} for a, b in zip(enc_outputs_class, enc_outputs_coord)
                ]
        # for global multi-label loss
        if gcs_transformer_out is not None:
            out["gcs_transformer_out"] = [{"pred_logits":gcs_transformer_out}]
        if gcs_back_out is not None:
            out["gcs_back_out"] = [{"pred_logits":gcs_back_out}]
        # for backbone loss
        if src_flatten_return is not None:
            backbone_coord = src_flatten_coord.sigmoid()
            backbone_class = self.transformer.backbone_out_class_embed(src_flatten_return)
            out['backbone_outputs'] = {'pred_logits': backbone_class, 'pred_boxes': backbone_coord}

        out['dn_meta'] = dn_meta

        out["topk_proposals"] = {"topk_proposals":topk_proposals, "level_start_index":level_start_index,\
                                  "spatial_shapes":spatial_shapes}

        # import os
        # if os.path.exists("./CAM"):
        #     import shutil
        #     shutil.rmtree("./CAM")
        # os.mkdir("./CAM")
        # self.CAM_VIS(memory, self.class_embed[5].state_dict(), targets, level_start_index, spatial_shapes, samples, self.transformer.enc_out_class_embed(hs_enc[-1]))


        # self.save_visual_query(level_start_index, spatial_shapes, topk_proposals, samples, targets, out)

        # final = out['pred_binary'].sigmoid()
        # self.save_his_img(final)


        # if not is_train:
        # #     enc = out['interm_outputs']['pred_binary'].sigmoid()
        # #     d0 = out['aux_outputs'][0]['pred_binary'].sigmoid()
        # #     d1 = out['aux_outputs'][1]['pred_binary'].sigmoid()
        # #     d2 = out['aux_outputs'][2]['pred_binary'].sigmoid()
        # #     d3 = out['aux_outputs'][3]['pred_binary'].sigmoid()
        # #     d4 = out['aux_outputs'][4]['pred_binary'].sigmoid()
        #     final = out['pred_binary'].sigmoid()
        # #     # final = d0 + d1 + d2 + d3 + d4 + final

        #     out_logit = out['pred_logits']
        # #     out_pred = out['pred_boxes']
        # #     # out['pred_logits'] = out_logit * (final >= 0.1)
        # #     # out['pred_boxes'] = out_pred * (final >= 0.1)

        #     out['pred_logits'] = out_logit * (final - final.min()) / (final.max() - final.min())
        # #     # out['pred_boxes'] = out_pred * final

        # #     # B, _, _ = final.shape
        # #     # for i in range(B):
        # #     #     tp = torch.where(final[i:i+1, :, 0] >= 0.1)            
        # #     #     out['pred_logits'][i:i+1, :, :] = torch.gather(out_logit[i:i+1, :, :], 1, tp[1].unsqueeze(-1).repeat(1, 1, 91))
        # #     #     out['pred_boxes'][i:i+1, :, :] = torch.gather(out_pred[i:i+1, :, :], 1, tp[1].unsqueeze(-1).repeat(1, 1, 4))

        return out

    def save_his_img(self, binary):
        b, n, c = binary.shape
        import matplotlib.pyplot as plt
        import numpy as np
        binary2 = binary.detach().cpu().numpy()
        for i in range(b):
            x = np.arange(n)

            plt.plot(x, binary2[i][:, 0])

            plt.xlabel('Query')

            # 添加y轴标签
            plt.ylabel('Probablity')
            # 添加图形标题
            plt.title('Probablity of Query')
            # 添加图例
            plt.legend()
            # 显示图形
            # plt.show()
            plt.savefig("his.png")
            plt.close()


    def get_cam(self, img, fea, weight_softmax):
        import numpy as np
        import cv2
        H, W, C = img.shape
        bz, nc, h, w = fea.shape        #1,960,7,7

        idx = 0

        cam = weight_softmax[idx].unsqueeze(0).mm(fea.contiguous().view((nc, h*w)))  #(5, 960) * (960, 7*7) -> (5, 7*7) （n,）是一个数组，既不是行向量也不是列向量
        cam = cam.view(h, w).permute(1, 0)
        mean_ = cam.mean()
        # cam_img = (cam - cam.min()) / (cam.max() - cam.min())  #Normalize

        std_ = cam.std()
        min_ = mean_ - 4 * std_
        max_ = mean_ + 4 * std_
        cam[cam < min_] = min_
        cam[cam > max_] = max_
        cam_img = (cam - min_) / (max_ - min_)


        cam_img = np.uint8(255 * cam_img.cpu().numpy())                      #Format as CV_8UC1 (as applyColorMap required)
        cam_re = cv2.resize(cam_img, (W, H))

        return cam_re


    def CAM_VIS(self, feature_conv, weight_softmax, targets, level_start_index, spatial_shapes,samples, preds):
        import numpy as np
        level0 = level_start_index[0].item()
        level1 = level_start_index[1].item()
        level2 = level_start_index[2].item()
        level3 = level_start_index[3].item()

        l1_h, l1_w = spatial_shapes[0][0], spatial_shapes[0][1]
        l2_h, l2_w = spatial_shapes[1][0], spatial_shapes[1][1]
        l3_h, l3_w = spatial_shapes[2][0], spatial_shapes[2][1]
        l4_h, l4_w = spatial_shapes[3][0], spatial_shapes[3][1]

        N, _, d = feature_conv.shape

        fea1 = feature_conv[:, level0:level1, :].view(N, l1_h, l1_w, d).permute(0, 3, 2, 1)
        fea2 = feature_conv[:, level1:level2, :].view(N, l2_h, l2_w, d).permute(0, 3, 2, 1)
        fea3 = feature_conv[:, level2:level3, :].view(N, l3_h, l3_w, d).permute(0, 3, 2, 1)
        fea4 = feature_conv[:, level3:, :].view(N, l4_h, l4_w, d).permute(0, 3, 1, 2)

        from util.visualizer import renorm
        import random, cv2
        img = renorm(samples.tensors.cpu()[0]).permute(1, 2, 0)
        img = np.uint8(img*255)
        img2 = img.copy()
        H, W, C = img2.shape

        # weight_softmax = torch.softmax(weight_softmax["weight"], dim=0)
        weight_softmax = weight_softmax["weight"]

        heat_map1 = self.get_cam(img, fea1, weight_softmax)
        heat_map2 = self.get_cam(img, fea2, weight_softmax)
        heat_map3 = self.get_cam(img, fea3, weight_softmax)
        heat_map4 = self.get_cam(img, fea4, weight_softmax)

        # bz, nc, h, w = fea1.shape        #1,960,7,7
        # output_cam = []
        # class_idx = 91
        # for idx in range(class_idx):  #只输出预测概率最大值结果不需要for循环
        #     max_, indce_ = preds.max(dim=2)
        #     idx = 50
        #     # idx = indce_[0][0]

        #     # feature_conv = fea1.reshape((nc, h*w))  # [960,7*7]
        #     # cam = weight_softmax[idx].dot(fea1.reshape((nc, h*w)))  #(5, 960) * (960, 7*7) -> (5, 7*7) （n,）是一个数组，既不是行向量也不是列向量
        #     cam = weight_softmax[idx].unsqueeze(0).mm(fea1.contiguous().view((nc, h*w)))  #(5, 960) * (960, 7*7) -> (5, 7*7) （n,）是一个数组，既不是行向量也不是列向量
        #     cam = cam.view(h, w).permute(1, 0)
        #     cam_img = (cam - cam.min()) / (cam.max() - cam.min())  #Normalize
        #     cam_img = np.uint8(255 * cam_img.cpu().numpy())                      #Format as CV_8UC1 (as applyColorMap required)
    
        #     cam_re = cv2.resize(cam_img, (W, H))
        #     output_cam.append(cam_re)  # Resize as image size
        #     # output_cam.append(cam_img)

        #     heat_map = cv2.applyColorMap(cam_re, cv2.COLORMAP_JET)
        #     # result = np.uint8(heat_map * 0.3 + img * 0.7)
                        
        #     result = cv2.addWeighted(img, 0.5, heat_map, 0.3, 0)
        #     result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite('./CAM/CAM_' + str(targets[0]["image_id"].item()) + "_" + str(idx) + '.jpg', result)
        #     break


        # cam = heat_map1 * 0.5 + heat_map2 * 0.5
        cam = heat_map1
        # cam_img = (cam - cam.min()) / (cam.max() - cam.min())  #Normalize
        cam_img = np.uint8(cam)                      #Format as CV_8UC1 (as applyColorMap required)

        

        heat_map = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)

        result = cv2.addWeighted(img, 0.5, heat_map, 0.5, 0)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite('./CAM/CAM_' + str(targets[0]["image_id"].item()) + "_" + '.jpg', result)



    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses, reweight, boost_loss):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.REWEIGHT = reweight
        if reweight:
            print(30*"===", "Using rewight for small objects!...")

        self.boost_loss = boost_loss

    def reweight_loss(self, targets_boxes):
        wh = targets_boxes[:, 2:]
        area_value = wh[:, 0] * wh[:, 1]

        # 原log方案
        # weight = (-torch.log(area_value) + 1e-5)    # 越小，相应的权重应该越大
        # norm = weight / torch.sum(weight)
        # new_weight = norm * targets_boxes.shape[0]

        # 指数方案，权重在0~1之间，不做归一化
        new_weight = torch.sqrt(area_value)
        new_weight = torch.exp(-new_weight) + 1.0

        # 固定权重值方案
        # new_weight = torch.ones_like(area_value)
        # new_weight[area_value < 0.01] = 3.0

        return new_weight


    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)    # 匹配到的box id
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        # target_classes_onehot[:,:,:1] = target_classes_onehot[:,:,-1:]    # ours
        target_classes_onehot = target_classes_onehot[:,:,:-1]

        if self.REWEIGHT:
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            new_weight = self.reweight_loss(target_boxes)
            loss_ce = sigmoid_focal_loss_reweight(src_logits, target_classes_onehot, num_boxes, new_weight, idx, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        else: 
            # 原方法
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
            
            # src_boxes = outputs['pred_boxes'][idx]
            # target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            # loss_ce = sigmoid_focal_loss_me(src_logits, target_classes_onehot, num_boxes, target_boxes, src_boxes, alpha=self.focal_alpha, gamma=2, idx=idx)
            
            if self.boost_loss:
                src_boxes = outputs['pred_boxes'][idx]
                target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
                loss_ce_box = sigmoid_focal_loss_small(src_logits, target_classes_onehot, num_boxes, target_boxes, src_boxes, alpha=self.focal_alpha, gamma=2, idx=idx) * src_logits.shape[1]

                # loss_ce = loss_ce + loss_ce_box
                loss_ce = loss_ce_box
                # loss_ce = 2 * loss_ce_box

            # alpha = 0.25; gamma= 2.0
            # out_prob = src_logits
            # tgt_ids = target_classes_o
            # out_prob = out_prob.flatten(0, 1).sigmoid() 
            # neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            # pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            # cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            # loss_ce = cost_class.mean(1).sum() / num_boxes
            losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_multi_label(self, outputs, targets):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        # target_classes_o = torch.stack([t["labels"] for t in targets], dim=0)
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        for i in range(len(targets)):
            lb = targets[i]["labels"]
            target_classes[i][lb] = lb

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1]],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(1, target_classes, 1)

        loss_bce = nn.BCEWithLogitsLoss()(src_logits, target_classes_onehot)
        losses = {'loss_bce': loss_bce}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def get_non_matched_loss(self, match_idx, outputs, target_boxes, num_boxes):
        B, N, _ = outputs['pred_wh'].shape

        # 未匹配到的损失
        tar_wh = torch.zeros_like(outputs['pred_wh']).cuda()
        pred_tp = torch.ones_like(outputs['pred_wh']).cuda()
        pred_tp[match_idx] = 0
        loss_bbox = F.l1_loss(outputs['pred_wh'], tar_wh, reduction='none')
        loss_bbox = loss_bbox * pred_tp
        
        loss = loss_bbox.sum() / (B*N - len(match_idx[1]))

        # 匹配到的损失
        src_boxes = outputs['pred_wh'][match_idx]
        loss_bbox = F.l1_loss(src_boxes, target_boxes[:, 2:], reduction='none')
        loss_bbox = loss_bbox.sum() / num_boxes

        return loss + 5 * loss_bbox

    def loss_wh_non_matched_boxes(self, outputs, indices, targets, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_wh' in outputs
        idx = self._get_src_permutation_idx(indices)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}

        loss_non_matched = self.get_non_matched_loss(idx, outputs, target_boxes, num_boxes)

        # 对非匹配框，w h 进行零值约束

        losses['loss_wh_unmatched'] = loss_non_matched

        return losses

    def loss_query_xy(self, outputs, num_boxes, targets):
        out_querys = outputs["topk_proposals"]

        topk_proposals = out_querys["topk_proposals"]
        level_start_index = out_querys["level_start_index"]
        spatial_shapes = out_querys["spatial_shapes"]

        level0 = level_start_index[0].item()
        level1 = level_start_index[1].item()
        level2 = level_start_index[2].item()
        level3 = level_start_index[3].item()

        l1_h, l1_w = spatial_shapes[0][0], spatial_shapes[0][1]
        l2_h, l2_w = spatial_shapes[1][0], spatial_shapes[1][1]
        l3_h, l3_w = spatial_shapes[2][0], spatial_shapes[2][1]
        l4_h, l4_w = spatial_shapes[3][0], spatial_shapes[3][1]

        shapes_ = {"l1_h":l1_h, "l1_w":l1_w, "l2_h":l2_h, "l2_w":l2_w,\
                   "l3_h":l3_h, "l3_w":l3_w, "l4_h":l4_h, "l4_w":l4_w}

        l1 = ((topk_proposals - level1) <= 0).int()
        l2 = ((topk_proposals - level2) <= 0).int()
        l2 = l2 - l1
        l3 = ((topk_proposals - level3) <= 0).int()
        l3 = l3 - l2 - l1
        l4 = (topk_proposals > level3).int()

        level_1_gt_h = ((l1 * topk_proposals) // l1_w) / l1_h
        level_1_gt_w = ((l1 * topk_proposals) % l1_w) / l1_w

        level_2_gt_h = ((l2 * (topk_proposals - level1)) // l2_w) / l2_h
        level_2_gt_w = ((l2 * (topk_proposals - level1)) % l2_w) / l2_w

        level_3_gt_h = ((l3 * (topk_proposals - level2)) // l3_w) / l3_h
        level_3_gt_w = ((l3 * (topk_proposals - level2)) % l3_w) / l3_w

        level_4_gt_h = ((l4 * (topk_proposals - level3)) // l4_w) / l4_h
        level_4_gt_w = ((l4 * (topk_proposals - level3)) % l4_w) / l4_w

        h_gt = level_1_gt_h + level_2_gt_h + level_3_gt_h + level_4_gt_h
        w_gt = level_1_gt_w + level_2_gt_w + level_3_gt_w + level_4_gt_w
        gt = torch.cat([w_gt.unsqueeze(-1), h_gt.unsqueeze(-1)], dim=2)

        l_dict = {}
        sz = targets[0]['size']
        if "pred_boxes" in outputs:
            bbox = outputs["pred_boxes"]
            new_bbox = self.process_wh(bbox[:, :, :2], l1, l2, l3, l4, sz, shapes_)
            loss_bbox = F.l1_loss(new_bbox, gt, reduction='none')
            l_dict["query_final"] = loss_bbox.sum() / num_boxes
        if "aux_outputs" in outputs:
            for i in range(len(outputs["aux_outputs"])):
                bbox = outputs["aux_outputs"][i]["pred_boxes"]
                new_bbox = self.process_wh(bbox[:, :, :2], l1, l2, l3, l4, sz, shapes_)
                loss_bbox = F.l1_loss(new_bbox, gt, reduction='none')
                l_dict['query_aux' + f'_{i}'] = loss_bbox.sum() / num_boxes
        if "interm_outputs" in outputs:
            if "pred_boxes" in outputs["interm_outputs"]:
                for i in range(len(outputs["interm_outputs"])):
                    bbox = outputs["interm_outputs"]["pred_boxes"]
                    new_bbox = self.process_wh(bbox[:, :, :2], l1, l2, l3, l4, sz, shapes_)
                    loss_bbox = F.l1_loss(new_bbox, gt, reduction='none')
                    l_dict['query_interm'] = loss_bbox.sum() / num_boxes
            
        return l_dict


    def process_wh(self, box, l1, l2, l3, l4, sz, shapes_):
        bl1_w = (box[:, :, 0] * l1) * sz[1] / 8 / shapes_["l1_w"]    # x
        bl1_h = (box[:, :, 1] * l1) * sz[0] / 8 / shapes_["l1_h"]# y

        bl2_w = (box[:, :, 0] * l2) * sz[1] / 16 / shapes_["l2_w"]# x
        bl2_h = (box[:, :, 1] * l2) * sz[0] / 16 / shapes_["l2_h"] # y

        bl3_w = (box[:, :, 0] * l3) * sz[1] / 32 / shapes_["l3_w"] # x
        bl3_h = (box[:, :, 1] * l3) * sz[0] / 32 / shapes_["l3_h"]# y

        bl4_w = (box[:, :, 0] * l4) * sz[1] / 64 / shapes_["l4_w"] # x
        bl4_h = (box[:, :, 1] * l4) * sz[0] / 64 / shapes_["l4_h"] # y

        ww = bl1_w + bl2_w + bl3_w + bl4_w
        hh = bl1_h + bl2_h + bl3_h + bl4_h
        ww = torch.cat([ww.unsqueeze(-1), hh.unsqueeze(-1)], dim=2)

        return ww

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        # loss_bbox[:, 0:1] = loss_bbox[:, 0:1] / target_boxes[:, 2:3]
        # loss_bbox[:, 1:2] = loss_bbox[:, 1:2] / target_boxes[:, 3:]
        # loss_bbox[:, 2:3] = loss_bbox[:, 2:3] / target_boxes[:, 2:3]
        # loss_bbox[:, 3:] = loss_bbox[:, 3:] / target_boxes[:, 3:]

        # loss_bbox[:, 0:1] = loss_bbox[:, 0:1] * torch.exp(-target_boxes[:, 2:3])
        # loss_bbox[:, 1:2] = loss_bbox[:, 1:2] * torch.exp(-target_boxes[:, 3:])
        # loss_bbox[:, 2:3] = loss_bbox[:, 2:3] * torch.exp(-target_boxes[:, 2:3])
        # loss_bbox[:, 3:] = loss_bbox[:, 3:] * torch.exp(-target_boxes[:, 3:])

        # loss_bbox[:, :2] = loss_bbox[:, :2] / target_boxes[:, 2:]
        # loss_bbox[:, 2:] = loss_bbox[:, 2:] / target_boxes[:, 2:]

        # gt = copy.copy(target_boxes)
        # pred_box = copy.copy(src_boxes)
        # loss_bbox2 = F.l1_loss(pred_box[:, :2] / gt[:, 2:], gt[:, :2] / gt[:, 2:], reduction='none')
        # loss_bbox[:, :2] = loss_bbox[:, :2] + loss_bbox2

        # 对非匹配框，w h 进行零值约束

        # 增加小目标权重
        if self.REWEIGHT:
            new_weight = self.reweight_loss(target_boxes)
            loss_bbox[:, :] = new_weight.unsqueeze(dim=1) * loss_bbox[:, :]

        losses = {}

        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        # test for siou
        # loss_giou = 1 - torch.diag(bbox_iou_my(box_ops.box_cxcywh_to_xyxy(src_boxes), \
        #                     box_ops.box_cxcywh_to_xyxy(target_boxes), x1y1x2y2=True, SIoU=True))

        if self.REWEIGHT:
            loss_giou = loss_giou * new_weight

        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
        with torch.no_grad():
            losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
            losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes


        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device=next(iter(outputs.values())).device
        indices = self.matcher(outputs_without_aux, targets)

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_ = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes_], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        # print(get_world_size())

        # Compute all the requested losses
        losses = {}

        # prepare for dn loss
        dn_meta = outputs['dn_meta']

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.range(0, len(targets[i]['labels']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']

            l_dict = {}
            for loss in self.losses:
                
                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,**kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)

            self.binary_dn_loss(output_known_lbs_bboxes, dn_pos_idx, num_boxes*scalar, losses)

            # self.wh_part_loss_dn(dn_pos_idx, output_known_lbs_bboxes, losses, targets, num_boxes*scalar)
        else:
            l_dict = dict()
            l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_binary_binary_dec_final_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_binary_dn_0'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_binary_dn_1'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_binary_dn_2'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_binary_dn_3'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_binary_dn_4'] = torch.as_tensor(0.).to('cuda')

            # l_dict['loss_wh_unmatched_final_wh_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_wh_unmatched_wh_dn_0'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_wh_unmatched_wh_dn_1'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_wh_unmatched_wh_dn_2'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_wh_unmatched_wh_dn_3'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_wh_unmatched_wh_dn_4'] = torch.as_tensor(0.).to('cuda')

            losses.update(l_dict)

        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                # for dn part
                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    l_dict={}
                    for loss in self.losses:
                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar,
                                                                 **kwargs))

                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                else:
                    l_dict = dict()
                    l_dict['loss_bbox_dn']=torch.as_tensor(0.).to('cuda')
                    l_dict['loss_giou_dn']=torch.as_tensor(0.).to('cuda')
                    l_dict['loss_ce_dn']=torch.as_tensor(0.).to('cuda')
                    l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            # interm_outputs = outputs['interm_outputs']
            # indices = self.matcher(interm_outputs, targets)
            # if return_indices:
            #     indices_list.append(indices)
            # for loss in self.losses:
            #     if loss == 'masks':
            #         # Intermediate masks losses are too costly to compute, we ignore them.
            #         continue
            #     kwargs = {}
            #     if loss == 'labels':
            #         # Logging is enabled only for the last layer
            #         kwargs = {'log': False}
            #     l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
            #     l_dict = {k + f'_interm': v for k, v in l_dict.items()}
            #     losses.update(l_dict)


            interm_outputs = outputs['interm_outputs']
            indices_interm = self.matcher(interm_outputs, targets)
            # indices_interm = indices
            if return_indices:
                indices_list.append(indices_interm)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, interm_outputs, targets, indices_interm, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # for gqs
        if 'interm_outputs_gqs' in outputs:
            interm_outputs = outputs['interm_outputs_gqs']
            indices_interm = self.matcher(interm_outputs, targets)
            if return_indices:
                indices_list.append(indices_interm)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}
                    # continue
                if loss == "cardinality":
                    continue
                l_dict = self.get_loss(loss, interm_outputs, targets, indices_interm, num_boxes, **kwargs)
                l_dict = {k + f'_interm_gqs': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # for backbone_outputs
        if 'backbone_outputs' in outputs:
            backbone_outputs = outputs['backbone_outputs']
            indices_interm = self.matcher(backbone_outputs, targets)
            if return_indices:
                indices_list.append(indices_interm)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}
                    # continue
                if loss == "cardinality":
                    continue
                l_dict = self.get_loss(loss, backbone_outputs, targets, indices_interm, num_boxes, **kwargs)
                l_dict = {k + f'_backbone': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # enc output loss
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # 匹配损失：encoder预测和最后的预测
        if False:
            interm_outputs = outputs['interm_outputs']
            decoder_out = [{"boxes": outputs_without_aux['pred_boxes'][i], "labels": torch.argmax(outputs_without_aux['pred_logits'][i], dim=1)} for i in range(len(outputs_without_aux['pred_boxes']))]
            indices = [torch.arange(0, 900)]
            indice = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(i, dtype=torch.int64)) for i in indices]
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}
                    # continue
                if loss == "cardinality":
                    continue
                l_dict = self.get_loss(loss, interm_outputs, decoder_out, indice, num_boxes, **kwargs)
                l_dict = {k + f'_match': v for k, v in l_dict.items()}
                losses.update(l_dict)

        if "gcs_transformer_out" in outputs:
            multi_label_loss = self.loss_multi_label(outputs=outputs["gcs_transformer_out"][0], targets=targets)
            l_dict = {k + f'_gcs_transformer': v for k, v in multi_label_loss.items()}
            losses.update(l_dict)

        if "gcs_back_out" in outputs:
            multi_label_enc_loss = self.loss_multi_label(outputs=outputs["gcs_back_out"][0], targets=targets)
            l_dict = {k + f'_gcs_backbone': v for k, v in multi_label_enc_loss.items()}
            losses.update(l_dict)

        wh_unmatched =False; UNMATCHED_ENCODER_LOSS = False
        # if wh_unmatched:
        #     self.wh_part_loss(outputs_without_aux, outputs, targets, num_boxes, losses)

        if UNMATCHED_ENCODER_LOSS:
            enc_out = outputs['interm_outputs']
            indices = self.matcher(enc_out, targets)
            unmatched_loss = self.loss_wh_non_matched_boxes(outputs=enc_out, indices=indices)
            l_dict = {k + f'_encoder_wh': v for k, v in unmatched_loss.items()}
            losses.update(l_dict)
        
        if 'pred_binary' in outputs_without_aux and num_boxes > 0:
            losses = self.binary_loss(outputs_without_aux, outputs, targets, num_boxes, losses)

        QUERY_LOSS = False    # 当前位置的query，中心预测在当前位置的约束
        if QUERY_LOSS:
            l_dict = self.loss_query_xy(outputs, num_boxes, targets)
            losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def wh_part_loss_dn(self, indices, outputs, losses, targets, num_boxes):
        if "pred_wh" in outputs:
            unmatched_loss = self.loss_wh_non_matched_boxes(outputs=outputs, indices=indices, targets=targets, num_boxes=num_boxes)
            l_dict = {k + f'_final_wh_dn': v for k, v in unmatched_loss.items()}
            losses.update(l_dict)

        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                unmatched_loss = self.loss_wh_non_matched_boxes(outputs=aux_outputs, indices=indices, targets=targets, num_boxes=num_boxes)
                l_dict = {k + f'_wh_dn_{idx}': v for k, v in unmatched_loss.items()}
                losses.update(l_dict)

    def wh_part_loss(self, indices, outputs, losses, targets, num_boxes):
        if "pred_wh" in outputs:
            unmatched_loss = self.loss_wh_non_matched_boxes(outputs=outputs, indices=indices, targets=targets, num_boxes=num_boxes)
            l_dict = {k + f'_final_wh': v for k, v in unmatched_loss.items()}
            losses.update(l_dict)

        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                unmatched_loss = self.loss_wh_non_matched_boxes(outputs=aux_outputs, indices=indices, targets=targets, num_boxes=num_boxes)
                l_dict = {k + f'_wh_{idx}': v for k, v in unmatched_loss.items()}
                losses.update(l_dict)

    def wh_part_loss(self, outputs_without_aux, outputs, targets, num_boxes, losses):
        B, N, _ = outputs['pred_wh'].shape

        COMBINE_ALL_INDICE = False
        if COMBINE_ALL_INDICE:
            indices_final = self.matcher(outputs_without_aux, targets)
            aux_0 = self.matcher(outputs['aux_outputs'][0], targets)
            aux_1 = self.matcher(outputs['aux_outputs'][1], targets)
            aux_2 = self.matcher(outputs['aux_outputs'][2], targets)
            aux_3 = self.matcher(outputs['aux_outputs'][3], targets)
            aux_4 = self.matcher(outputs['aux_outputs'][4], targets)
            # indices_enc = self.matcher(outputs['interm_outputs'], targets)
            indices = []

            for i in range(B):
                a = torch.cat([indices_final[i][0], aux_0[i][0], aux_1[i][0], \
                                           aux_2[i][0], aux_3[i][0], aux_4[i][0]], dim=0)
                b = torch.cat([indices_final[i][1], aux_0[i][1], aux_1[i][1], \
                                           aux_2[i][1], aux_3[i][1], aux_4[i][1]], dim=0)
                c = torch.cat([a.unsqueeze(0), b.unsqueeze(0)], dim=0)
                indices.append(c)

        if 'pred_wh' in outputs_without_aux:
            if not COMBINE_ALL_INDICE:
                indices = self.matcher(outputs_without_aux, targets)
            
            unmatched_loss = self.loss_wh_non_matched_boxes(outputs=outputs, indices=indices, targets=targets, num_boxes=num_boxes)
            l_dict = {k + f'_final_wh': v for k, v in unmatched_loss.items()}
            losses.update(l_dict)

        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                if not COMBINE_ALL_INDICE:
                    indices = self.matcher(aux_outputs, targets)
                unmatched_loss = self.loss_wh_non_matched_boxes(outputs=aux_outputs, indices=indices, targets=targets, num_boxes=num_boxes)
                l_dict = {k + f'_wh_{idx}': v for k, v in unmatched_loss.items()}
                losses.update(l_dict)

        return losses

    def binary_dn_loss(self, output_known_lbs_bboxes, indices, num_boxes, losses):
        assert 'pred_binary' in output_known_lbs_bboxes

        if "pred_binary" in output_known_lbs_bboxes:
            B, N, _ = output_known_lbs_bboxes['pred_binary'].shape

            if 'pred_binary' in output_known_lbs_bboxes:
                dec_final = self.loss_dynamic_binary_labels(outputs=output_known_lbs_bboxes, targets=None, indices=indices, num_boxes=num_boxes)
                l_dict = {k + f'_binary_dec_final_dn': v for k, v in dec_final.items()}
                losses.update(l_dict)
            if 'aux_outputs' in output_known_lbs_bboxes:
                for idx, aux_outputs in enumerate(output_known_lbs_bboxes['aux_outputs']):
                    multi_layer_dec_loss = self.loss_dynamic_binary_labels(outputs=aux_outputs, targets=None, indices=indices, num_boxes=num_boxes)
                    l_dict = {k + f'_dn_{idx}': v for k, v in multi_layer_dec_loss.items()}
                    losses.update(l_dict)

        return losses


    def binary_loss(self, outputs_without_aux, outputs, targets, num_boxes, losses):
        B, N, _ = outputs['pred_logits'].shape

        COMBINE_ALL_INDICE = False
        if COMBINE_ALL_INDICE:
            indices_final = self.matcher(outputs_without_aux, targets)
            aux_0 = self.matcher(outputs['aux_outputs'][0], targets)
            aux_1 = self.matcher(outputs['aux_outputs'][1], targets)
            aux_2 = self.matcher(outputs['aux_outputs'][2], targets)
            aux_3 = self.matcher(outputs['aux_outputs'][3], targets)
            aux_4 = self.matcher(outputs['aux_outputs'][4], targets)
            # indices_enc = self.matcher(outputs['interm_outputs'], targets)
            indices = []

            for i in range(B):
                a = torch.cat([indices_final[i][0], aux_0[i][0], aux_1[i][0], \
                                           aux_2[i][0], aux_3[i][0], aux_4[i][0]], dim=0)
                b = torch.cat([indices_final[i][1], aux_0[i][1], aux_1[i][1], \
                                           aux_2[i][1], aux_3[i][1], aux_4[i][1]], dim=0)
                c = torch.cat([a.unsqueeze(0), b.unsqueeze(0)], dim=0)
                indices.append(c)

        if 'pred_binary' in outputs_without_aux:
            if not COMBINE_ALL_INDICE:
                indices = self.matcher(outputs_without_aux, targets)
            dec_final = self.loss_dynamic_binary_labels(outputs=outputs, targets=targets, indices=indices, num_boxes=num_boxes)
            l_dict = {k + f'_binary_dec_final': v for k, v in dec_final.items()}
            losses.update(l_dict)
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                if not COMBINE_ALL_INDICE:
                    indices = self.matcher(aux_outputs, targets)
                multi_layer_dec_loss = self.loss_dynamic_binary_labels(outputs=outputs['aux_outputs'][idx], targets=targets, indices=indices, num_boxes=num_boxes)
                l_dict = {k + f'_{idx}': v for k, v in multi_layer_dec_loss.items()}
                losses.update(l_dict)
        if 'interm_outputs' in outputs:
            # if not COMBINE_ALL_INDICE:
            #     indices = self.matcher(outputs['interm_outputs'], targets)
            indices = self.matcher(outputs['interm_outputs'], targets)
            dec_final = self.loss_dynamic_binary_labels(outputs=outputs['interm_outputs'], targets=targets, indices=indices, num_boxes=num_boxes)
            # a = outputs['interm_outputs']['pred_binary'].sigmoid() > 0.5
            # print(a.sum())
            l_dict = {k + f'_binary_enc': v for k, v in dec_final.items()}
            losses.update(l_dict)
        return losses

    def loss_dynamic_binary_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_binary' in outputs
        src_logits = outputs["pred_binary"]

        idx = self._get_src_permutation_idx(indices)    # 匹配到的box id
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=src_logits.dtype, device=src_logits.device)
        target_classes[idx] = 1

        # target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], 2],
        #                                     dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        # target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)  # [b, n, 2] 0  负样本， 1正样本

        # 原方法
        loss_ce = sigmoid_focal_loss(src_logits, target_classes.unsqueeze(-1), num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
 
        losses = {'loss_binary': loss_ce}

        return losses



    def prep_for_dn(self,dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups==0
        single_pad=pad_size//num_dn_groups

        return output_known_lbs_bboxes,single_pad,num_dn_groups


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100, nms_iou_threshold=-1) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        # num_select = self.num_select
        num_select = outputs['pred_logits'].shape[1]
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # ******* 剔除预测为no object的目标 ************
        # a = torch.argmax(out_logits, dim=2) 
        # b = torch.where(a == 0)
        # out_logits[b] = -1000000.0
        # ******* 剔除预测为no object的目标 ************

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        if test:
            assert not not_to_xyxy
            boxes[:,:,2:] = boxes[:,:,2:] - boxes[:,:,:2]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if self.nms_iou_threshold > 0:
            item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold) for b,s in zip(boxes, scores)]

            results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i]} for s, l, b, i in zip(scores, labels, boxes, item_indices)]
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


@MODULE_BUILD_FUNCS.registe_with_name(module_name='dino')
def build_dino(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    # if args.dataset_file == "coco_panoptic":
    #     # for panoptic, we just add a num_classes that is large enough to hold
    #     # max_obj_id + 1, but the exact value doesn't really matter
    #     num_classes = 250
    # if args.dataset_file == 'o365':
    #     num_classes = 366
    # if args.dataset_file == 'vanke':
    #     num_classes = 51
    num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)

    if not args.only_decoder:
        transformer = build_deformable_transformer(args)
    elif args.only_decoder and not args.use_gqs_transformer:
        transformer = build_deformable_decoder_transformer(args)
        print("We only use decoder transformer!...")

    

    try:
        match_unstable_error = args.match_unstable_error
        dn_labelbook_size = args.dn_labelbook_size
    except:
        match_unstable_error = True
        dn_labelbook_size = num_classes

    try:
        dec_pred_class_embed_share = args.dec_pred_class_embed_share
    except:
        dec_pred_class_embed_share = True
    try:
        dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    except:
        dec_pred_bbox_embed_share = True

    model = DINO(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=args.random_refpoints_xy,
        fix_refpoints_hw=args.fix_refpoints_hw,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_class_embed_share=dec_pred_class_embed_share,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        # two stage
        two_stage_type=args.two_stage_type,
        # box_share
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        decoder_sa_type=args.decoder_sa_type,
        num_patterns=args.num_patterns,
        dn_number = args.dn_number if args.use_dn else 0,
        dn_box_noise_scale = args.dn_box_noise_scale,
        dn_label_noise_ratio = args.dn_label_noise_ratio,
        dn_labelbook_size = dn_labelbook_size,
        use_gqs_transformer=args.use_gqs_transformer,
        nms_decoder=args.nms_decoder,
        binary = args.binary_loss,
        wh_part = args.unmatch_wh,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)

    # prepare weight dict
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    
    # for DN training
    if args.use_dn:
        weight_dict['loss_ce_dn'] = args.cls_loss_coef
        weight_dict['loss_bbox_dn'] = args.bbox_loss_coef
        weight_dict['loss_giou_dn'] = args.giou_loss_coef

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    clean_weight_dict = copy.deepcopy(weight_dict)

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != 'no':
        interm_weight_dict = {}
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        _coeff_weight_dict = {
            'loss_ce': 1.0,
            'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
        }
        try:
            interm_loss_coef = args.interm_loss_coef
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update({k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items()})
        weight_dict.update(interm_weight_dict)

    if args.use_xywh_loss:
        xywh_weight_dict = {}
        xywh_dict = {"loss_xy":1.0, "loss_xy_dn":1.0, "loss_hw":1.0, "loss_hw_dn":1.0}
        weight_dict.update(xywh_dict)
        weight_dict["loss_xy_interm"] = 1.0
        weight_dict["loss_hw_interm"] = 1.0

        for i in range(args.dec_layers - 1):
            xywh_weight_dict.update({k + f'_{i}': v for k, v in xywh_dict.items()})
        weight_dict.update(xywh_weight_dict)

    if args.use_gqs_transformer:
        weight_dict["loss_ce_interm_gqs"] = 1.0
        weight_dict["loss_bbox_interm_gqs"] = 5.0
        weight_dict["loss_giou_interm_gqs"] = 2.0
        weight_dict["loss_xy_interm_gqs"] = 1.0
        weight_dict["loss_hw_interm_gqs"] = 1.0
    if args.gcs_enc:
        weight_dict["loss_bce_gcs_backbone"] = 1.0
    if args.gcs_transformer_enc:
        weight_dict["loss_bce_gcs_transformer"] = 1.0
    if args.loss_backbone:
        weight_dict["loss_ce_backbone"] = 1.0
        weight_dict["loss_bbox_backbone"] = 5.0
        weight_dict["loss_giou_backbone"] = 2.0
        weight_dict["loss_xy_backbone"] = 1.0
        weight_dict["loss_hw_backbone"] = 1.0
    if args.match:
        weight_dict["loss_ce_match"] = 1.0
        weight_dict["loss_bbox_match"] = 1.0
        weight_dict["loss_giou_match"] = 1.0
        weight_dict["loss_xy_match"] = 1.0
        weight_dict["loss_hw_match"] = 1.0
    if args.unmatch_wh:
        weight_dict["loss_bbox_unmatched_wh"] = 1.0
    if args.unmatch_encoder_wh:
        weight_dict["loss_bbox_unmatched_encoder_wh"] = 5.0
    if args.binary_loss:
        weight_dict["loss_binary_binary_dec_final"] = 1.0
        weight_dict["loss_binary_0"] = 1.0
        weight_dict["loss_binary_1"] = 1.0
        weight_dict["loss_binary_2"] = 1.0
        weight_dict["loss_binary_3"] = 1.0
        weight_dict["loss_binary_4"] = 1.0
        weight_dict["loss_binary_binary_dec_final_dn"] = 1.0
        weight_dict["loss_binary_dn_0"] = 1.0
        weight_dict["loss_binary_dn_1"] = 1.0
        weight_dict["loss_binary_dn_2"] = 1.0
        weight_dict["loss_binary_dn_3"] = 1.0
        weight_dict["loss_binary_dn_4"] = 1.0

        weight_dict["loss_binary_binary_enc"] = 2.0

    if args.query_loss:
        cc = 0.5
        weight_dict['query_final'] = cc
        weight_dict['query_aux_0'] = cc
        weight_dict['query_aux_1'] = cc
        weight_dict['query_aux_2'] = cc
        weight_dict['query_aux_3'] = cc
        weight_dict['query_aux_4'] = cc
        weight_dict['query_interm'] = cc

    if args.unmatch_wh:
        wwhh = 0.0
        weight_dict['loss_wh_unmatched_final_wh_dn'] = wwhh
        weight_dict['loss_wh_unmatched_wh_dn_0'] = wwhh
        weight_dict['loss_wh_unmatched_wh_dn_1'] = wwhh
        weight_dict['loss_wh_unmatched_wh_dn_2'] = wwhh
        weight_dict['loss_wh_unmatched_wh_dn_3'] = wwhh
        weight_dict['loss_wh_unmatched_wh_dn_4'] = wwhh

        weight_dict['loss_wh_unmatched_final_wh'] = wwhh
        weight_dict['loss_wh_unmatched_wh_0'] = wwhh
        weight_dict['loss_wh_unmatched_wh_1'] = wwhh
        weight_dict['loss_wh_unmatched_wh_2'] = wwhh
        weight_dict['loss_wh_unmatched_wh_3'] = wwhh
        weight_dict['loss_wh_unmatched_wh_4'] = wwhh


    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses,
                             reweight=args.reweight_small,
                             boost_loss=args.boost_loss)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(num_select=args.num_select, nms_iou_threshold=args.nms_iou_threshold)}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
