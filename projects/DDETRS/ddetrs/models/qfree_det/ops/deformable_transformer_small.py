# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR Transformer class.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import math, random
import copy
from typing import Optional

import torch
from torch import nn, Tensor

from torchvision.ops.boxes import nms, batched_nms

from util import box_ops

from util.misc import inverse_sigmoid
from .utils import gen_encoder_output_proposals, MLP,_get_activation_fn, gen_sineembed_for_position
from .ops.modules import MSDeformAttn
# from models.dino.cspn import Affinity_Propagate
# from models.dino.deformable_fusion_transformer import TransformerEncoderFusion, DeformableTransformerEncoderLayer_Fusion

import torch.nn.functional as F

from models.dino.cross_fusion import Cross_Fusion

from models.dino.external_attention import External_attention, External_attention_mh


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, groups=mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, groups=mid_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DeformableTransformer(nn.Module):

    def __init__(self, d_model=256, nhead=8, 
                 num_queries=300, 
                 num_encoder_layers=6,
                 num_unicoder_layers=0,
                 num_decoder_layers=6, 
                 dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=4,
                 num_patterns=0,
                 modulate_hw_attn=False,
                 # for deformable encoder
                 deformable_encoder=False,
                 deformable_decoder=False,
                 num_feature_levels=1,
                 enc_n_points=4,
                 dec_n_points=4,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 # init query
                 learnable_tgt_init=False,
                 decoder_query_perturber=None,
                 add_channel_attention=False,
                 add_pos_value=False,
                 random_refpoints_xy=False,
                 # two stage
                 two_stage_type='no', # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
                 two_stage_pat_embed=0,
                 two_stage_add_query_num=0,
                 two_stage_learn_wh=False,
                 two_stage_keep_all_tokens=False,
                 # evo of #anchors
                 dec_layer_number=None,
                 rm_enc_query_scale=True,
                 rm_dec_query_scale=True,
                 rm_self_attn_layers=None,
                 key_aware_type=None,
                 # layer share
                 layer_share_type=None,
                 # for detach
                 rm_detach=None,
                 decoder_sa_type='ca', 
                 module_seq=['sa', 'ca', 'ffn'],
                 # for dn
                 embed_init_tgt=False,

                 use_detached_boxes_dec_out=False,

                gcs_enc = False,
                gcs_transformer_enc = False,
                gqs_use = False,

                cross_coding = False,
                nms_free = 0.5,
                nms_decoder = 0,
                class_num = 1,
                binary = False,
                 ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.deformable_encoder = deformable_encoder
        self.deformable_decoder = deformable_decoder
        self.two_stage_keep_all_tokens = two_stage_keep_all_tokens
        self.num_queries = num_queries
        self.random_refpoints_xy = random_refpoints_xy
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out
        self.gcs_transformer_enc = gcs_transformer_enc
        self.gcs_enc = gcs_enc
        self.gqs_use = gqs_use
        self.nms_th_free = nms_free
        self.nms_decoder = nms_decoder
        self.num_class = class_num
        self.binary = binary

        self.cross_coding = cross_coding
        if self.gqs_use:
            print("Using graph-based query selection method!...")

        assert query_dim == 4

        if num_feature_levels > 1:
            assert deformable_encoder, "only support deformable_encoder for num_feature_levels > 1"
        if use_deformable_box_attn:
            assert deformable_encoder or deformable_encoder

        assert layer_share_type in [None, 'encoder', 'decoder', 'both']
        if layer_share_type in ['encoder', 'both']:
            enc_layer_share = True
        else:
            enc_layer_share = False
        if layer_share_type in ['decoder', 'both']:
            dec_layer_share = True
        else:
            dec_layer_share = False
        assert layer_share_type is None

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']

        # choose encoder layer type
        if deformable_encoder:
            encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points, add_channel_attention=add_channel_attention, use_deformable_box_attn=use_deformable_box_attn, box_attn_type=box_attn_type)
        else:
            raise NotImplementedError
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, 
            encoder_norm, d_model=d_model, 
            num_queries=num_queries,
            deformable_encoder=deformable_encoder, 
            enc_layer_share=enc_layer_share, 
            two_stage_type=two_stage_type
        )

        # choose decoder layer type
        if deformable_decoder:
            decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points, use_deformable_box_attn=use_deformable_box_attn, box_attn_type=box_attn_type,
                                                          key_aware_type=key_aware_type,
                                                          decoder_sa_type=decoder_sa_type,
                                                          module_seq=module_seq)

        else:
            raise NotImplementedError

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                        return_intermediate=return_intermediate_dec,
                                        d_model=d_model, query_dim=query_dim, 
                                        modulate_hw_attn=modulate_hw_attn,
                                        num_feature_levels=num_feature_levels,
                                        deformable_decoder=deformable_decoder,
                                        decoder_query_perturber=decoder_query_perturber, 
                                        dec_layer_number=dec_layer_number, rm_dec_query_scale=rm_dec_query_scale,
                                        dec_layer_share=dec_layer_share,
                                        use_detached_boxes_dec_out=use_detached_boxes_dec_out,
                                        nms_decoder=nms_decoder,
                                        binary = binary,
                                        )

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries  # useful for single stage model only
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0

        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
            else:
                self.level_embed = None
        
        self.learnable_tgt_init = learnable_tgt_init
        assert learnable_tgt_init, "why not learnable_tgt_init"
        self.embed_init_tgt = embed_init_tgt
        if (two_stage_type != 'no' and embed_init_tgt) or (two_stage_type == 'no'):
            self.tgt_embed = nn.Embedding(self.num_queries, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.tgt_embed = None
            # self.head_tgt_bias = MLP(d_model, d_model*8, d_model, 2)    # 为tgt做条件处理
            # self.bias_norm = nn.LayerNorm(d_model)
            # self.tgt_bias_box = MLP(d_model, d_model, 4, 3)
            # self.bias_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)

            # self.activation_gelu = nn.ReLU()
    
        #================================ for fusion of small object detection ===========================#
        # encoder_layer_fusion = DeformableTransformerEncoderLayer_Fusion(d_model, dim_feedforward,
        #                                                   dropout, activation,
        #                                                   num_feature_levels, nhead, enc_n_points, add_channel_attention=add_channel_attention, use_deformable_box_attn=use_deformable_box_attn, box_attn_type=box_attn_type)
        # encoder_norm_fusion = nn.LayerNorm(d_model) if normalize_before else None
        # num_encoder_fusion_layers = 2

        if self.cross_coding:
            self.fusion = Cross_Fusion()
            print("Using Cross Coding Moudle...")

        #================================ for fusion of small object detection ===========================#

        # for two stage
        self.two_stage_type = two_stage_type
        self.two_stage_pat_embed = two_stage_pat_embed
        self.two_stage_add_query_num = two_stage_add_query_num
        self.two_stage_learn_wh = two_stage_learn_wh
        assert two_stage_type in ['no', 'standard'], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type =='standard':
            # anchor selection at the output of encoder
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)      
            
            if two_stage_pat_embed > 0:
                self.pat_embed_for_2stage = nn.Parameter(torch.Tensor(two_stage_pat_embed, d_model))
                nn.init.normal_(self.pat_embed_for_2stage)

            if two_stage_add_query_num > 0:
                self.tgt_embed = nn.Embedding(self.two_stage_add_query_num, d_model)

            if two_stage_learn_wh:

                self.two_stage_wh_embedding = nn.Embedding(1, 2)
            else:
                self.two_stage_wh_embedding = None

        if two_stage_type == 'no':
            self.init_ref_points(num_queries) # init self.refpoint_embed

        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None
        self.enc_binary_class = None
        self.conditional_binary_enc = None

        self.enc_exsa = None
        # self.enc_exsa = nn.Linear(256, 256)
        # self.enc_exsa = External_attention_mh(256)
        # self.enc_norm = nn.LayerNorm(256)
        
        # self.enc_mlp = None

        # self.conv1 = nn.Sequential(nn.Conv2d(d_model, d_model, 3, 1, 1, bias=False, groups=d_model))
        # self.conv2 = nn.Conv2d(d_model, d_model, 3, 1, 1, bias=False, groups=d_model)
        # self.conv3 = nn.Conv2d(d_model, d_model, 3, 1, 1, bias=False, groups=d_model)
        # self.conv4 = nn.Conv2d(d_model, d_model, 3, 1, 1, bias=False, groups=d_model)

        # self.conv1 = nn.Sequential(nn.Conv2d(d_model, d_model, 3, 1, 1, bias=False, groups=d_model), \
        #                            nn.Conv2d(d_model, d_model, 3, 1, 1, bias=False, groups=d_model))
        # self.conv2 = nn.Sequential(nn.Conv2d(d_model, d_model, 3, 1, 1, bias=False, groups=d_model), \
        #                            nn.Conv2d(d_model, d_model, 3, 1, 1, bias=False, groups=d_model))
        # self.conv3 = nn.Sequential(nn.Conv2d(d_model, d_model, 3, 1, 1, bias=False, groups=d_model), \
        #                            nn.Conv2d(d_model, d_model, 3, 1, 1, bias=False, groups=d_model))
        # self.conv4 = nn.Sequential(nn.Conv2d(d_model, d_model, 3, 1, 1, bias=False, groups=d_model), \
        #                            nn.Conv2d(d_model, d_model, 3, 1, 1, bias=False, groups=d_model))

        # self.conv1 = DoubleConv(d_model, d_model)


        # process for backbone
        # self.backbone_process = nn.Conv2d(d_model, d_model, 1, 1, 0, bias=False, groups=1)
        # self.backbone_norm = nn.LayerNorm(d_model)
        # self.backbone_fuse = nn.Linear(d_model*2, d_model)
        self.backbone_out_class_embed = None
        self.backbone_out_bbox_embed = None

        if self.gcs_transformer_enc:
            self.class_fuse = nn.Sequential(nn.Linear(256*4, 91))
            print("Use GCS in transformer encoder!...")
        if self.gcs_enc:
            self.class_fuse_enc = nn.Sequential(nn.Linear(256, 91))
            print("Use GCS in backbone!...")

        # evolution of anchors
        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            if self.two_stage_type != 'no' or num_patterns == 0:
                assert dec_layer_number[0] == num_queries, f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries})"
            else:
                assert dec_layer_number[0] == num_queries * num_patterns, f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries}) * num_patterns({num_patterns})"

        self._reset_parameters()

        self.rm_self_attn_layers = rm_self_attn_layers
        if rm_self_attn_layers is not None:
            print("Removing the self-attn in {} decoder layers".format(rm_self_attn_layers))
            for lid, dec_layer in enumerate(self.decoder.layers):
                if lid in rm_self_attn_layers:
                    dec_layer.rm_self_attn_modules()

        self.rm_detach = rm_detach
        if self.rm_detach:
            assert isinstance(rm_detach, list)
            assert any([i in ['enc_ref', 'enc_tgt', 'dec'] for i in rm_detach])
        self.decoder.rm_detach = rm_detach

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

        if self.two_stage_learn_wh:
            nn.init.constant_(self.two_stage_wh_embedding.weight, math.log(0.05 / (1 - 0.05)))

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)
        
        if self.random_refpoints_xy:

            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

    def save_quries_distribution_pos(self, level_start_index, indices, value, topk, spatial_shapes, name):
        import matplotlib.pyplot as plt
        import numpy as np
        
        confidence, topk_proposals = value[:, :topk], indices[:, :topk]

        confidence_sigmoid = torch.sigmoid(confidence)

        level0 = level_start_index[0].item()
        level1 = level_start_index[1].item()
        level2 = level_start_index[2].item()
        level3 = level_start_index[3].item()

        l1_h, l1_w = spatial_shapes[0][0], spatial_shapes[0][1]
        l2_h, l2_w = spatial_shapes[1][0], spatial_shapes[1][1]
        l3_h, l3_w = spatial_shapes[2][0], spatial_shapes[2][1]
        l4_h, l4_w = spatial_shapes[3][0], spatial_shapes[3][1]

        n, num = topk_proposals.shape
        lv1 = torch.zeros((l1_h, l1_w), dtype=torch.long)
        lv2 = torch.zeros((l2_h, l2_w), dtype=torch.long)
        lv3 = torch.zeros((l3_h, l3_w), dtype=torch.long)
        lv4 = torch.zeros((l4_h, l4_w), dtype=torch.long)
        for i in range(num):
            id = topk_proposals[0, i].item()
            val = confidence_sigmoid[0, i].item()
            val = val * 4
            if id >= level0 and id < level1:                
                hi = id // l1_w    # 行
                wj = id % l1_w     # 列
                lv1[hi.item(), wj.item()] = int(255 * val)
            elif id >= level1 and id < level2:
                id = id - level1
                hi = id // l2_w    # 行
                wj = id % l2_w     # 列
                lv2[hi.item(), wj.item()] = int(255 * val)
            elif id >= level2 and id < level3:
                id = id - level2
                hi = id // l3_w    # 行
                wj = id % l3_w     # 列
                lv3[hi.item(), wj.item()] = int(255 * val)
            elif id >= level3:
                id = id - level3
                hi = id // l4_w    # 行
                wj = id % l4_w     # 列
                lv4[hi.item(), wj.item()] = int(255 * val)

        import cv2
        lv1 = cv2.applyColorMap(cv2.convertScaleAbs(lv1.cpu().numpy(), alpha=1), cv2.COLORMAP_JET)
        lv2 = cv2.applyColorMap(cv2.convertScaleAbs(lv2.cpu().numpy(), alpha=1), cv2.COLORMAP_JET)
        lv3 = cv2.applyColorMap(cv2.convertScaleAbs(lv3.cpu().numpy(), alpha=1), cv2.COLORMAP_JET)
        lv4 = cv2.applyColorMap(cv2.convertScaleAbs(lv4.cpu().numpy(), alpha=1), cv2.COLORMAP_JET)
        cv2.imwrite("./out_fig/queries_" + str(name) + "_" + str(topk) + "_lv1.png", lv1)
        cv2.imwrite("./out_fig/queries_" + str(name) + "_" + str(topk) + "_lv2.png", lv2)
        cv2.imwrite("./out_fig/queries_" + str(name) + "_" + str(topk) + "_lv3.png", lv3)
        cv2.imwrite("./out_fig/queries_" + str(name) + "_" + str(topk) + "_lv4.png", lv4)

        x = topk_proposals[0, :].cpu().numpy()
        y = confidence.cpu().detach().numpy()[0,:]
        plt.plot(x, y, '.')
        plt.xlabel('Queries')
        plt.scatter(level_start_index.cpu().numpy(), [-2,-2,-2,-2], c="red")

        # 添加y轴标签
        plt.ylabel('Pixel Range')
        # 添加图形标题
        plt.title('Disturibution of the Selected Quries ' + str(topk))
        # 添加图例
        plt.legend()
        # 显示图形
        plt.savefig("./out_fig/queries_" + str(name) + "_" + str(topk) + ".png")
        plt.clf()

    def save_quries_distribution(self, level_start_index, enc_outputs_class_unselected, topk, spatial_shapes, name, value_):
        import matplotlib.pyplot as plt
        import numpy as np
        
        confidence, topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)

        value_ = torch.sigmoid(value_)

        level0 = level_start_index[0].item()
        level1 = level_start_index[1].item()
        level2 = level_start_index[2].item()
        level3 = level_start_index[3].item()

        l1_h, l1_w = spatial_shapes[0][0], spatial_shapes[0][1]
        l2_h, l2_w = spatial_shapes[1][0], spatial_shapes[1][1]
        l3_h, l3_w = spatial_shapes[2][0], spatial_shapes[2][1]
        l4_h, l4_w = spatial_shapes[3][0], spatial_shapes[3][1]

        n, num = topk_proposals.shape
        lv1 = torch.zeros((l1_h, l1_w), dtype=torch.long)
        lv2 = torch.zeros((l2_h, l2_w), dtype=torch.long)
        lv3 = torch.zeros((l3_h, l3_w), dtype=torch.long)
        lv4 = torch.zeros((l4_h, l4_w), dtype=torch.long)

        l0 = 0; l1 = 0; l2 = 0; l3 = 0
        for i in range(num):
            id = topk_proposals[0, i].item()
            val = value_[0, i].item()
            if id >= level0 and id < level1:                
                hi = id // l1_w    # 行
                wj = id % l1_w     # 列
                lv1[hi.item(), wj.item()] = int(255 * val + 100)
                l0 = l0 + 1
            elif id >= level1 and id < level2:
                id = id - level1
                hi = id // l2_w    # 行
                wj = id % l2_w     # 列
                lv2[hi.item(), wj.item()] = int(255 * val + 100)
                l1 = l1 + 1
            elif id >= level2 and id < level3:
                id = id - level2
                hi = id // l3_w    # 行
                wj = id % l3_w     # 列
                lv3[hi.item(), wj.item()] = int(255 * val + 100)
                l2 = l2 + 1
            elif id >= level3:
                id = id - level3
                hi = id // l4_w    # 行
                wj = id % l4_w     # 列
                lv4[hi.item(), wj.item()] = int(255 * val + 100)
                l3 = l3 + 1
        print("number: ", l0, l1, l2, l3)

        import cv2
        lv1 = cv2.applyColorMap(cv2.convertScaleAbs(lv1.cpu().numpy(), alpha=1), cv2.COLORMAP_JET)
        lv2 = cv2.applyColorMap(cv2.convertScaleAbs(lv2.cpu().numpy(), alpha=1), cv2.COLORMAP_JET)
        lv3 = cv2.applyColorMap(cv2.convertScaleAbs(lv3.cpu().numpy(), alpha=1), cv2.COLORMAP_JET)
        lv4 = cv2.applyColorMap(cv2.convertScaleAbs(lv4.cpu().numpy(), alpha=1), cv2.COLORMAP_JET)
        cv2.imwrite("./out_fig/queries_" + str(name) + "_" + str(topk) + "_lv1.png", lv1)
        cv2.imwrite("./out_fig/queries_" + str(name) + "_" + str(topk) + "_lv2.png", lv2)
        cv2.imwrite("./out_fig/queries_" + str(name) + "_" + str(topk) + "_lv3.png", lv3)
        cv2.imwrite("./out_fig/queries_" + str(name) + "_" + str(topk) + "_lv4.png", lv4)


        x = topk_proposals[0, :].cpu().numpy()
        y = confidence.cpu().detach().numpy()[0,:]
        plt.plot(x, y, '.')
        plt.xlabel('Queries')
        plt.scatter(level_start_index.cpu().numpy(), [-2,-2,-2,-2], c="red")

        # 添加y轴标签
        plt.ylabel('Pixel Range')
        # 添加图形标题
        plt.title('Disturibution of the Selected Quries ' + str(topk))
        # 添加图例
        plt.legend()
        # 显示图形
        plt.savefig("./out_fig/queries_" + str(name) + "_" + str(topk) + ".png")
        plt.clf()


    def select_query_with_position_2(self, level_start_index, enc_outputs_class_unselected, spatial_shapes, topk, num=2):
        level0 = level_start_index[0].item()
        level1 = level_start_index[1].item()
        level2 = level_start_index[2].item()
        level3 = level_start_index[3].item()
    
        # 获取topk
        topk_value, topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk*num, dim=1) # bs, nq

        # 排序
        id_sort_value, sort_indice = torch.sort(topk_proposals, dim=1)
        p_value_sort_origin = torch.gather(topk_value, 1, sort_indice)

        # 每个batch中不同图像的索引不同
        N, _, _ = enc_outputs_class_unselected.shape
        temp = torch.zeros((N, 3), dtype=torch.long)
        for i in range(N):
            value1 = torch.where(id_sort_value[i, :] >= level1)
            value2 = torch.where(id_sort_value[i, :] >= level2)
            value3 = torch.where(id_sort_value[i, :] >= level3)
            if len(value1[0]) > 0:
                temp[i, 0] = value1[0][0] + 1
            else:
                temp[i, 0] = topk*num    # 全部在第一层

            if len(value1[0]) > 0 and len(value2[0]) > 0:
                temp[i, 1] = value2[0][0] + 1
            else:  # 下一层没有
                temp[i, 1] = topk*num

            if len(value2[0]) > 0 and len(value3[0]) > 0:
                temp[i, 2] = value3[0][0] + 1
            else:
                temp[i, 2] = topk*num


        dis = None
        for i in range(N):
            tp = None
            if temp[i, 0] > 0:
                q1 = id_sort_value[i:i+1, :temp[i, 0]]
                p1 = p_value_sort_origin[i:i+1, :temp[i, 0]]                
                res = self.get_distance_metric(q1, spatial_shapes[0, 0].item(), spatial_shapes[0, 1].item(), threshold=16)
                dis1 = torch.sigmoid(p1) + 0.5*(1.0 / (res + 1.0))
                tp = dis1
            if temp[i, 1] > 0:
                q2 = id_sort_value[i:i+1, temp[i, 0]:temp[i, 1]] - level1
                res = self.get_distance_metric(q2, spatial_shapes[1, 0].item(), spatial_shapes[1, 1].item(), threshold=8)
                p2 = p_value_sort_origin[i:i+1, temp[i, 0]:temp[i, 1]]
                dis2 = torch.sigmoid(p2) +  0.5*(1.0 / (res + 1.0))
                if tp is None:
                    tp = dis2
                else:
                    tp = torch.cat([tp, dis2], dim=1)
            if temp[i, 2] > 0:
                q3 = id_sort_value[i:i+1, temp[i, 1]:temp[i, 2]] - level2
                res = self.get_distance_metric(q3, spatial_shapes[2, 0].item(), spatial_shapes[2, 1].item(), threshold=4)
                p3 = p_value_sort_origin[i:i+1, temp[i, 1]:temp[i, 2]]
                dis3 = torch.sigmoid(p3) +  0.5*(1.0 / (res + 1.0))
                if tp is None:
                    tp = dis3
                else:
                    tp = torch.cat([tp, dis3], dim=1)

            if temp[i, 2] > 0:
                q4 = id_sort_value[i:i+1, temp[i, 2]:] - level3
                res = self.get_distance_metric(q4, spatial_shapes[3, 0].item(), spatial_shapes[3, 1].item(), threshold=2)
                p4 = p_value_sort_origin[i:i+1, temp[i, 2]:]
                dis4 = torch.sigmoid(p4) +  0.5*(1.0 / (res + 1.0))

                if tp is None:
                    tp = dis4
                else:
                    tp = torch.cat([tp, dis4], dim=1)   

            if dis is None:
                dis = tp
            else:
                dis = torch.cat([dis, tp], dim=0)

        # 对全局的距离矩阵进行排序
        topk_v, topk_ps = torch.topk(dis, topk, dim=1) # bs, nq

        # 同时考虑置信度
        # topk_v, topk_ps = torch.topk(dis*torch.sigmoid(value_sort_origin), topk, dim=1) # bs, nq

        # 挑选原序列中的indice
        indice_sort = torch.gather(id_sort_value, 1, topk_ps)    # sort_value与dis的顺序相同
        value_sort = torch.gather(p_value_sort_origin, 1, topk_ps)    # sort_value与dis的顺序相同

        return indice_sort, value_sort


    # pos 优化向量表示
    def select_query_with_position_3(self, level_start_index, enc_outputs_class_unselected, spatial_shapes, topk, num=2):
        level0 = level_start_index[0].item()
        level1 = level_start_index[1].item()
        level2 = level_start_index[2].item()
        level3 = level_start_index[3].item()
    
        # 获取topk
        topk_value, topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk*num, dim=1) # bs, nq

        # 排序
        id_sort_value, sort_indice = torch.sort(topk_proposals, dim=1)
        p_value_sort_origin = torch.gather(topk_value, 1, sort_indice)

        # 每个batch中不同图像的索引不同
        N, _, _ = enc_outputs_class_unselected.shape
        temp = torch.zeros((N, 3), dtype=torch.long)
        for i in range(N):
            value1 = torch.where(id_sort_value[i, :] >= level1)
            value2 = torch.where(id_sort_value[i, :] >= level2)
            value3 = torch.where(id_sort_value[i, :] >= level3)
            if len(value1[0]) > 0:
                temp[i, 0] = value1[0][0] + 1
            else:
                temp[i, 0] = topk*num    # 全部在第一层

            if len(value1[0]) > 0 and len(value2[0]) > 0:
                temp[i, 1] = value2[0][0] + 1
            else:  # 下一层没有
                temp[i, 1] = topk*num

            if len(value2[0]) > 0 and len(value3[0]) > 0:
                temp[i, 2] = value3[0][0] + 1
            else:
                temp[i, 1] = topk*num


        dis = None
        for i in range(N):
            tp = None
            if temp[i, 0] > 0:
                q1 = id_sort_value[i:i+1, :temp[i, 0]]
                p1 = p_value_sort_origin[i:i+1, :temp[i, 0]]                
                # res, c = self.get_distance_metric(q1, spatial_shapes[0, 0].item(), spatial_shapes[0, 1].item(), threshold=8)
                out = self.get_distance_metric(q1, spatial_shapes[0, 0].item(), spatial_shapes[0, 1].item(), threshold=8)
                # dis1 = torch.sigmoid(p1) + 0.5*(1.0 / (res + 1.0))
                # dis1 = p1@c.float() / res + 1 / res
                dis1 = p1@out
                tp = dis1

            if temp[i, 1] > 0:
                q2 = id_sort_value[i:i+1, temp[i, 0]:temp[i, 1]] - level1
                out = self.get_distance_metric(q2, spatial_shapes[1, 0].item(), spatial_shapes[1, 1].item(), threshold=4)
                p2 = p_value_sort_origin[i:i+1, temp[i, 0]:temp[i, 1]]
                # dis2 = torch.sigmoid(p2) +  0.5*(1.0 / (res + 1.0))
                dis2 = p2@out
                if tp is None:
                    tp = dis2
                else:
                    tp = torch.cat([tp, dis2], dim=1)
            if temp[i, 2] > 0:
                q3 = id_sort_value[i:i+1, temp[i, 1]:temp[i, 2]] - level2
                out = self.get_distance_metric(q3, spatial_shapes[2, 0].item(), spatial_shapes[2, 1].item(), threshold=2)
                p3 = p_value_sort_origin[i:i+1, temp[i, 1]:temp[i, 2]]
                # dis3 = torch.sigmoid(p3) +  0.5*(1.0 / (res + 1.0))
                dis3 = p3@out
                if tp is None:
                    tp = dis3
                else:
                    tp = torch.cat([tp, dis3], dim=1)

            if temp[i, 2] > 0:
                q4 = id_sort_value[i:i+1, temp[i, 2]:] - level3
                out = self.get_distance_metric(q4, spatial_shapes[3, 0].item(), spatial_shapes[3, 1].item(), threshold=2)
                p4 = p_value_sort_origin[i:i+1, temp[i, 2]:]
                # dis4 = torch.sigmoid(p4) +  0.5*(1.0 / (res + 1.0))
                dis4 = p4@out

                if tp is None:
                    tp = dis4
                else:
                    tp = torch.cat([tp, dis4], dim=1) 

            if dis is None:
                dis = tp
            else:
                dis = torch.cat([dis, tp], dim=0)

        # 对全局的距离矩阵进行排序
        topk_v, topk_ps = torch.topk(dis, topk, dim=1) # bs, nq

        # 同时考虑置信度
        # topk_v, topk_ps = torch.topk(dis*torch.sigmoid(value_sort_origin), topk, dim=1) # bs, nq

        # 挑选原序列中的indice
        indice_sort = torch.gather(id_sort_value, 1, topk_ps)    # sort_value与dis的顺序相同
        value_sort = torch.gather(p_value_sort_origin, 1, topk_ps)    # sort_value与dis的顺序相同

        return indice_sort, value_sort

    def select_query_with_position_only_firstlayer(self, level_start_index, enc_outputs_class_unselected, spatial_shapes, topk, num=2):
        level0 = level_start_index[0].item()
        level1 = level_start_index[1].item()
        level2 = level_start_index[2].item()
        level3 = level_start_index[3].item()
    
        # 获取topk
        topk_value, topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk*num, dim=1) # bs, nq

        # 排序
        id_sort_value, sort_indice = torch.sort(topk_proposals, dim=1)
        p_value_sort_origin = torch.gather(topk_value, 1, sort_indice)

        # 每个batch中不同图像的索引不同
        N, _, _ = enc_outputs_class_unselected.shape
        temp = torch.zeros((N, 1), dtype=torch.long)
        for i in range(N):
            value1 = torch.where(id_sort_value[i, :] >= level1)
            if len(value1[0]) > 0:
                temp[i, 0] = value1[0][0] + 1
            else:
                temp[i, 0] = topk*num    # 全部在第一层

        tops = None
        for i in range(N):
            if temp[i, 0] > 0:
                q1 = id_sort_value[i:i+1, :temp[i, 0]]
                p1 = p_value_sort_origin[i:i+1, :temp[i, 0]]                
                res = self.get_distance_metric(q1, spatial_shapes[0, 0].item(), spatial_shapes[0, 1].item(), threshold=8)
                dis = torch.sigmoid(p1) + 0.5*(1.0 / (res + 1.0))

                # origin: 
                q = id_sort_value[i:i+1, temp[i, 0]:]
                topkv, top_ko = torch.topk(q, topk - dis.shape[1] // 2, dim=1)

                # 对全局的距离矩阵进行排序
                topk_v, topk_ps = torch.topk(dis, dis.shape[1] // 2, dim=1) # bs, nq

                indices = torch.cat([top_ko, topk_ps], dim=1)

                if tops == None:
                    tops = indices
                else:
                    tops = torch.cat([tops, indices], dim=0)


        # 挑选原序列中的indice
 
        indice_sort = torch.gather(id_sort_value, 1, tops)    # sort_value与dis的顺序相同
        value_sort = torch.gather(p_value_sort_origin, 1, tops)    # sort_value与dis的顺序相同

        return indice_sort, value_sort


    def select_query_with_position_6(self, level_start_index, output_memory, enc_outputs_class_unselected, spatial_shapes, topk, num=2):
        level0 = level_start_index[0].item()
        level1 = level_start_index[1].item()
        level2 = level_start_index[2].item()
        level3 = level_start_index[3].item()
    
        # 获取topk
        topk_value_2, topk_proposals_2 = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk*num, dim=1) # bs, nq
        topk_data = torch.gather(output_memory, 1, topk_proposals_2.unsqueeze(dim=2).repeat(1, 1, 
                                                    output_memory.shape[2]))

        # 排序
        id_sort_value, sort_indice = torch.sort(topk_proposals_2, dim=1)
        p_value_sort_origin = torch.gather(topk_value_2, 1, sort_indice)
        topk_data_sort = torch.gather(topk_data, 1, sort_indice.unsqueeze(dim=2).repeat(1, 1, 
                                                    output_memory.shape[2]))    # 排序后的数据

        # 每个batch中不同图像的索引不同
        N, _, _ = enc_outputs_class_unselected.shape
        temp = torch.zeros((N, 3), dtype=torch.long)
        for i in range(N):
            value1 = torch.where(id_sort_value[i, :] >= level1)
            value2 = torch.where(id_sort_value[i, :] >= level2)
            value3 = torch.where(id_sort_value[i, :] >= level3)
            if len(value1[0]) > 0:
                temp[i, 0] = value1[0][0] + 1
            else:
                temp[i, 0] = topk*num    # 全部在第一层

            if len(value1[0]) > 0 and len(value2[0]) > 0:
                temp[i, 1] = value2[0][0] + 1
            else:  # 下一层没有
                temp[i, 1] = topk*num

            if len(value2[0]) > 0 and len(value3[0]) > 0:
                temp[i, 2] = value3[0][0] + 1
            else:
                temp[i, 1] = topk*num


        dis = None
        for i in range(N):
            tp = None
            if temp[i, 0] > 0:
                q1 = id_sort_value[i:i+1, :temp[i, 0]]
                p1 = p_value_sort_origin[i:i+1, :temp[i, 0]]                
                # f1 = topk_data_sort[i:i+1, :temp[i, 0], :]
                # res, c = self.get_distance_metric(q1, spatial_shapes[0, 0].item(), spatial_shapes[0, 1].item(), threshold=8)
                out = self.get_distance_metric(q1, p1, spatial_shapes[0, 0].item(), spatial_shapes[0, 1].item(), threshold=1)
                # dis1 = torch.sigmoid(p1) + 0.5*(1.0 / (res + 1.0))
                # dis1 = p1@c.float() / res + 1 / res

                # dis1 = p1.permute(0, 2, 1)@out
                # tp = dis1.permute(0, 2, 1)

                dis1 = torch.sigmoid(p1) * out
                tp = dis1


            if temp[i, 1] > 0:
                q2 = id_sort_value[i:i+1, temp[i, 0]:temp[i, 1]] - level1
                p2 = p_value_sort_origin[i:i+1, temp[i, 0]:temp[i, 1]]
                out = self.get_distance_metric(q2, p2, spatial_shapes[1, 0].item(), spatial_shapes[1, 1].item(), threshold=2)

                # dis2 = torch.sigmoid(p2) +  0.5*(1.0 / (res + 1.0))

                dis2 = torch.sigmoid(p2) * out #  p2.permute(0, 2, 1)@out
                if tp is None:
                    tp = dis2  #.permute(0, 2, 1)
                else:
                    # tp = torch.cat([tp, dis2.permute(0, 2, 1)], dim=1)
                    tp = torch.cat([tp, dis2], dim=1)


            if temp[i, 2] > 0:
                q3 = id_sort_value[i:i+1, temp[i, 1]:temp[i, 2]] - level2
                p3 = p_value_sort_origin[i:i+1, temp[i, 1]:temp[i, 2]]
                out = self.get_distance_metric(q3, p3, spatial_shapes[2, 0].item(), spatial_shapes[2, 1].item(), threshold=4)
                
                # dis3 = torch.sigmoid(p3) +  0.5*(1.0 / (res + 1.0))
                dis3 = torch.sigmoid(p3) * out # p3.permute(0, 2, 1)@out
                if tp is None:
                    tp = dis3 #.permute(0, 2, 1)
                else:
                    # tp = torch.cat([tp, dis3.permute(0, 2, 1)], dim=1)
                    tp = torch.cat([tp, dis3], dim=1)

            if temp[i, 2] > 0:
                q4 = id_sort_value[i:i+1, temp[i, 2]:] - level3
                p4 = p_value_sort_origin[i:i+1, temp[i, 2]:]
                out = self.get_distance_metric(q4, p4, spatial_shapes[3, 0].item(), spatial_shapes[3, 1].item(), threshold=8)
                
                # dis4 = torch.sigmoid(p4) +  0.5*(1.0 / (res + 1.0))
                dis4 = torch.sigmoid(p4) * out # p4.permute(0, 2, 1)@out

                if tp is None:
                    tp = dis4   # .permute(0, 2, 1)
                else:
                    # tp = torch.cat([tp, dis4.permute(0, 2, 1)], dim=1)
                    tp = torch.cat([tp, dis4], dim=1) 

            if dis is None:
                dis = tp
            else:
                dis = torch.cat([dis, tp], dim=0)

        # gcn
        # if self.gcn_embed is not None:
        #     dis = self.gcn_embed(dis)


        # 对全局的距离矩阵进行排序
        # out = self.enc_out_class_gcn_embed(dis)
        # topk_v, topk_ps = torch.topk(out.max(-1)[0], topk, dim=1) # bs, nq
        topk_v, topk_ps = torch.topk(dis, topk, dim=1) # bs, nq
        # data_topk_v = torch.gather(dis, 1, topk_ps.unsqueeze(dim=2).repeat(1, 1, 
        #                                             dis.shape[2]))    # 更新之后的特征
        data_topk_v = torch.gather(topk_data_sort, 1, topk_ps.unsqueeze(dim=2).repeat(1, 1, 
                                                    topk_data_sort.shape[2]))    # 更新之后的特征

        # 挑选原序列中的indice
        indice_sort = torch.gather(id_sort_value, 1, topk_ps)    # sort_value与dis的顺序相同
        value_sort = torch.gather(p_value_sort_origin, 1, topk_ps)    # sort_value与dis的顺序相同

        return indice_sort, value_sort, data_topk_v, topk_value_2, topk_proposals_2
    

    def select_query_with_position_only_firstlayer(self, level_start_index, enc_outputs_class_unselected, spatial_shapes, topk, num=2):
        level0 = level_start_index[0].item()
        level1 = level_start_index[1].item()
        level2 = level_start_index[2].item()
        level3 = level_start_index[3].item()
    
        # 获取topk
        topk_value, topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk*num, dim=1) # bs, nq

        # 排序
        id_sort_value, sort_indice = torch.sort(topk_proposals, dim=1)
        p_value_sort_origin = torch.gather(topk_value, 1, sort_indice)

        # 每个batch中不同图像的索引不同
        N, _, _ = enc_outputs_class_unselected.shape
        temp = torch.zeros((N, 1), dtype=torch.long)
        for i in range(N):
            value1 = torch.where(id_sort_value[i, :] >= level1)
            if len(value1[0]) > 0:
                temp[i, 0] = value1[0][0] + 1
            else:
                temp[i, 0] = topk*num    # 全部在第一层

        tops = None
        for i in range(N):
            if temp[i, 0] > 0:
                q1 = id_sort_value[i:i+1, :temp[i, 0]]
                p1 = p_value_sort_origin[i:i+1, :temp[i, 0]]                
                res = self.get_distance_metric(q1, spatial_shapes[0, 0].item(), spatial_shapes[0, 1].item(), threshold=8)
                dis = torch.sigmoid(p1) + 0.5*(1.0 / (res + 1.0))

                # origin: 
                q = id_sort_value[i:i+1, temp[i, 0]:]
                topkv, top_ko = torch.topk(q, topk - dis.shape[1] // 2, dim=1)

                # 对全局的距离矩阵进行排序
                topk_v, topk_ps = torch.topk(dis, dis.shape[1] // 2, dim=1) # bs, nq

                indices = torch.cat([top_ko, topk_ps], dim=1)

                if tops == None:
                    tops = indices
                else:
                    tops = torch.cat([tops, indices], dim=0)


        # 挑选原序列中的indice
 
        indice_sort = torch.gather(id_sort_value, 1, tops)    # sort_value与dis的顺序相同
        value_sort = torch.gather(p_value_sort_origin, 1, tops)    # sort_value与dis的顺序相同

        return indice_sort, value_sort

    def select_query_with_position_4(self, level_start_index, output_memory, enc_outputs_class_unselected, spatial_shapes, topk, num=2):
        level0 = level_start_index[0].item()
        level1 = level_start_index[1].item()
        level2 = level_start_index[2].item()
        level3 = level_start_index[3].item()
    
        # 获取topk
        topk_value_2, topk_proposals_2 = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk*num, dim=1) # bs, nq
        topk_data = torch.gather(output_memory, 1, topk_proposals_2.unsqueeze(dim=2).repeat(1, 1, 
                                                    output_memory.shape[2]))

        # 排序
        id_sort_value, sort_indice = torch.sort(topk_proposals_2, dim=1)
        p_value_sort_origin = torch.gather(topk_value_2, 1, sort_indice)
        topk_data_sort = torch.gather(topk_data, 1, sort_indice.unsqueeze(dim=2).repeat(1, 1, 
                                                    output_memory.shape[2]))    # 排序后的数据

        # 每个batch中不同图像的索引不同
        N, _, _ = enc_outputs_class_unselected.shape
        temp = torch.zeros((N, 3), dtype=torch.long)
        for i in range(N):
            value1 = torch.where(id_sort_value[i, :] >= level1)
            value2 = torch.where(id_sort_value[i, :] >= level2)
            value3 = torch.where(id_sort_value[i, :] >= level3)
            if len(value1[0]) > 0:
                temp[i, 0] = value1[0][0] + 1
            else:
                temp[i, 0] = topk*num    # 全部在第一层

            if len(value1[0]) > 0 and len(value2[0]) > 0:
                temp[i, 1] = value2[0][0] + 1
            else:  # 下一层没有
                temp[i, 1] = topk*num

            if len(value2[0]) > 0 and len(value3[0]) > 0:
                temp[i, 2] = value3[0][0] + 1
            else:
                temp[i, 1] = topk*num


        dis = None
        for i in range(N):
            tp = None
            if temp[i, 0] > 0:
                q1 = id_sort_value[i:i+1, :temp[i, 0]]
                p1 = p_value_sort_origin[i:i+1, :temp[i, 0]]                
                # f1 = topk_data_sort[i:i+1, :temp[i, 0], :]
                # res, c = self.get_distance_metric(q1, spatial_shapes[0, 0].item(), spatial_shapes[0, 1].item(), threshold=8)
                out = self.get_distance_metric(q1, p1, spatial_shapes[0, 0].item(), spatial_shapes[0, 1].item(), threshold=8)
                # dis1 = torch.sigmoid(p1) + 0.5*(1.0 / (res + 1.0))
                # dis1 = p1@c.float() / res + 1 / res

                # dis1 = p1.permute(0, 2, 1)@out
                # tp = dis1.permute(0, 2, 1)

                dis1 = torch.sigmoid(p1) * (1 - out)
                # dis1 = torch.sigmoid(p1) * out
                # dis1 = torch.sigmoid(p1)
                # dis1 = 1 - out.unsqueeze(0)
                tp = dis1


            if temp[i, 1] > 0:
                q2 = id_sort_value[i:i+1, temp[i, 0]:temp[i, 1]] - level1
                p2 = p_value_sort_origin[i:i+1, temp[i, 0]:temp[i, 1]]
                out = self.get_distance_metric(q2, p2, spatial_shapes[1, 0].item(), spatial_shapes[1, 1].item(), threshold=4)

                dis2 = torch.sigmoid(p2) * (1 - out)
                # dis2 = torch.sigmoid(p2) * out
                # dis2 = torch.sigmoid(p2)
                # dis2 = 1 - out.unsqueeze(0)
                if tp is None:
                    tp = dis2  #.permute(0, 2, 1)
                else:
                    # tp = torch.cat([tp, dis2.permute(0, 2, 1)], dim=1)
                    tp = torch.cat([tp, dis2], dim=1)


            if temp[i, 2] > 0:
                q3 = id_sort_value[i:i+1, temp[i, 1]:temp[i, 2]] - level2
                p3 = p_value_sort_origin[i:i+1, temp[i, 1]:temp[i, 2]]
                out = self.get_distance_metric(q3, p3, spatial_shapes[2, 0].item(), spatial_shapes[2, 1].item(), threshold=2)
                
                dis3 = torch.sigmoid(p3) * (1 - out) 
                # dis3 = torch.sigmoid(p3) * out
                # dis3 = torch.sigmoid(p3)
                # dis3 = 1 - out.unsqueeze(0)
                if tp is None:
                    tp = dis3 #.permute(0, 2, 1)
                else:
                    # tp = torch.cat([tp, dis3.permute(0, 2, 1)], dim=1)
                    tp = torch.cat([tp, dis3], dim=1)

            if temp[i, 2] > 0:
                q4 = id_sort_value[i:i+1, temp[i, 2]:] - level3
                p4 = p_value_sort_origin[i:i+1, temp[i, 2]:]
                out = self.get_distance_metric(q4, p4, spatial_shapes[3, 0].item(), spatial_shapes[3, 1].item(), threshold=2)
                
                dis4 = torch.sigmoid(p4) * (1 - out)
                # dis4 = torch.sigmoid(p4) * out
                # dis4 = torch.sigmoid(p4)
                # dis4 = 1 - out.unsqueeze(0)

                if tp is None:
                    tp = dis4   # .permute(0, 2, 1)
                else:
                    # tp = torch.cat([tp, dis4.permute(0, 2, 1)], dim=1)
                    tp = torch.cat([tp, dis4], dim=1) 

            if dis is None:
                dis = tp
            else:
                dis = torch.cat([dis, tp], dim=0)

        # gcn
        # if self.gcn_embed is not None:
        #     dis = self.gcn_embed(dis)


        # 对全局的距离矩阵进行排序
        # out = self.enc_out_class_gcn_embed(dis)
        # topk_v, topk_ps = torch.topk(out.max(-1)[0], topk, dim=1) # bs, nq
        topk_v, topk_ps = torch.topk(dis, topk, dim=1) # bs, nq
        # data_topk_v = torch.gather(dis, 1, topk_ps.unsqueeze(dim=2).repeat(1, 1, 
        #                                             dis.shape[2]))    # 更新之后的特征
        data_topk_v = torch.gather(topk_data_sort, 1, topk_ps.unsqueeze(dim=2).repeat(1, 1, 
                                                    topk_data_sort.shape[2]))    # 更新之后的特征

        # 挑选原序列中的indice
        indice_sort = torch.gather(id_sort_value, 1, topk_ps)    # sort_value与dis的顺序相同
        value_sort = torch.gather(p_value_sort_origin, 1, topk_ps)    # sort_value与dis的顺序相同

        return indice_sort, value_sort
    


    def select_query_with_position_5(self, level_start_index, output_memory, enc_outputs_class_unselected, spatial_shapes, topk, num=2):
        level0 = level_start_index[0].item()
        level1 = level_start_index[1].item()
        level2 = level_start_index[2].item()
        level3 = level_start_index[3].item()
    
        # 获取topk, 从topk中查找第一层
        topk = 400; level0_k = 500
        topk_value_2, topk_proposals_2 = torch.topk(enc_outputs_class_unselected[:, level1:, :].max(-1)[0], topk, dim=1) # bs, nq
        # topk_data = torch.gather(output_memory, 1, topk_proposals_2.unsqueeze(dim=2).repeat(1, 1, 
        #                                             output_memory.shape[2]))
        topk_proposals_2 = topk_proposals_2 + level1

        # 排序
        id_sort_value, sort_indice = torch.sort(topk_proposals_2, dim=1)
        p_value_sort_origin = torch.gather(topk_value_2, 1, sort_indice)
        # topk_data_sort = torch.gather(topk_data, 1, sort_indice.unsqueeze(dim=2).repeat(1, 1, 
        #                                             output_memory.shape[2]))    # 排序后的数据

        # 每个batch中不同图像的索引不同
        N, _, _ = enc_outputs_class_unselected.shape
        temp = torch.zeros((N, 3), dtype=torch.long)
        out_indice = None; out_value = None
        for i in range(N):
            value1 = torch.where(id_sort_value[i, :] >= level1)
            value2 = torch.where(id_sort_value[i, :] >= level2)
            value3 = torch.where(id_sort_value[i, :] >= level3)
            # if len(value1[0]) > 0:
            #     temp[i, 0] = value1[0][0] + 1    # 记录起始点
            # else:
            #     temp[i, 0] = topk*num    # 全部在第一层

            # if len(value1[0]) > 0 and len(value2[0]) > 0:
            #     temp[i, 1] = value2[0][0] + 1
            # else:  # 下一层没有
            #     temp[i, 1] = topk*num

            # if len(value2[0]) > 0 and len(value3[0]) > 0:
            #     temp[i, 2] = value3[0][0] + 1
            # else:
            #     temp[i, 1] = topk*num

            num_query_1 = len(value1[0])
            num_query_0 = topk - num_query_1
            num_query_0 = level0_k
            num_candidate = 2
            # print(num_query_0, len(value1[0])-len(value2[0]), len(value2[0])-len(value3[0]), len(value3[0]))

            # 第一层
            topk_value_level0, topk_proposals_level0 = torch.topk(enc_outputs_class_unselected[i:i+1, :level1, :].max(-1)[0], num_query_0*num_candidate, dim=1) # bs, nq
            # topk_value_level0, topk_proposals_level0 = torch.topk(torch.mean(torch.topk(enc_outputs_class_unselected[i:i+1, :level1, :], 5, dim=2)[0], dim=2), num_query_0*num_candidate, dim=1) # bs, nq

            # if temp[i, 0] > 0:
            if True:
                # q1 = id_sort_value[i:i+1, :temp[i, 0]]
                # p1 = p_value_sort_origin[i:i+1, :temp[i, 0]]
                q1 = topk_proposals_level0
                p1 = topk_value_level0
                out = self.get_distance_metric(q1, p1, spatial_shapes[0, 0].item(), spatial_shapes[0, 1].item(), threshold=8)
                # dis1 = torch.sigmoid(p1) + 0.5*(1.0 / (res + 1.0))
                # dis1 = p1@c.float() / res + 1 / res

                # dis1 = p1.permute(0, 2, 1)@out
                # tp = dis1.permute(0, 2, 1)

                dis1 = torch.sigmoid(p1) * (1 - out)
                # dis1 = torch.sigmoid(p1) * out
                # dis1 = torch.sigmoid(p1)
                # dis1 = 1 - out.unsqueeze(0)
                # dis1 = out.unsqueeze(0)


            # if temp[i, 1] > 0:
            #     q2 = id_sort_value[i:i+1, temp[i, 0]:temp[i, 1]] - level1
            #     p2 = p_value_sort_origin[i:i+1, temp[i, 0]:temp[i, 1]]
            #     out = self.get_distance_metric(q2, p2, spatial_shapes[1, 0].item(), spatial_shapes[1, 1].item(), threshold=2)

            #     # dis2 = torch.sigmoid(p2) * (1 - out) #  p2.permute(0, 2, 1)@out
            #     # dis2 = torch.sigmoid(p2)
            #     dis2 = 1 - out.unsqueeze(0)
            #     if tp is None:
            #         tp = dis2  #.permute(0, 2, 1)
            #     else:
            #         # tp = torch.cat([tp, dis2.permute(0, 2, 1)], dim=1)
            #         tp = torch.cat([tp, dis2], dim=1)


            # if temp[i, 2] > 0:
            #     q3 = id_sort_value[i:i+1, temp[i, 1]:temp[i, 2]] - level2
            #     p3 = p_value_sort_origin[i:i+1, temp[i, 1]:temp[i, 2]]
            #     out = self.get_distance_metric(q3, p3, spatial_shapes[2, 0].item(), spatial_shapes[2, 1].item(), threshold=4)
                
            #     # dis3 = torch.sigmoid(p3) * (1 - out) # p3.permute(0, 2, 1)@out
            #     # dis3 = torch.sigmoid(p3)
            #     dis3 = 1 - out.unsqueeze(0)
            #     if tp is None:
            #         tp = dis3 #.permute(0, 2, 1)
            #     else:
            #         # tp = torch.cat([tp, dis3.permute(0, 2, 1)], dim=1)
            #         tp = torch.cat([tp, dis3], dim=1)

            # if temp[i, 2] > 0:
            #     q4 = id_sort_value[i:i+1, temp[i, 2]:] - level3
            #     p4 = p_value_sort_origin[i:i+1, temp[i, 2]:]
            #     out = self.get_distance_metric(q4, p4, spatial_shapes[3, 0].item(), spatial_shapes[3, 1].item(), threshold=8)
                
            #     # dis4 = torch.sigmoid(p4) * (1 - out) # p4.permute(0, 2, 1)@out
            #     # dis4 = torch.sigmoid(p4)
            #     dis4 = 1 - out.unsqueeze(0)

            #     if tp is None:
            #         tp = dis4   # .permute(0, 2, 1)
            #     else:
            #         # tp = torch.cat([tp, dis4.permute(0, 2, 1)], dim=1)
            #         tp = torch.cat([tp, dis4], dim=1) 


            # 对全局的距离矩阵进行排序
            # out = self.enc_out_class_gcn_embed(dis)
            # topk_v, topk_ps = torch.topk(out.max(-1)[0], topk, dim=1) # bs, nq
            topk_v, topk_ps = torch.topk(dis1, num_query_0, dim=1) # bs, nq
            # data_topk_v = torch.gather(dis, 1, topk_ps.unsqueeze(dim=2).repeat(1, 1, 
            #                                             dis.shape[2]))    # 更新之后的特征


            # 挑选原序列中的indice
            indice_sort = torch.gather(topk_proposals_level0, 1, topk_ps)    # sort_value与dis的顺序相同
            value_sort = torch.gather(topk_value_level0, 1, topk_ps)    # sort_value与dis的顺序相同

            indice_sort = torch.cat([indice_sort, id_sort_value[i:i+1, num_query_0:]], dim=1)
            value_sort = torch.cat([value_sort, p_value_sort_origin[i:i+1, num_query_0:]], dim=1)

            # indice_sort = torch.cat([indice_sort, id_sort_value[i:i+1, :]], dim=1)
            # value_sort = torch.cat([value_sort, p_value_sort_origin[i:i+1, :]], dim=1)



            if out_indice is None:
                out_indice = indice_sort
                out_value = value_sort
            else:
                out_indice = torch.cat([out_indice, indice_sort], dim=0)
                out_value = torch.cat([out_value, value_sort], dim=0)

        return out_indice, out_value


    def select_query_with_position_6(self, level_start_index, output_memory, enc_outputs_class_unselected, spatial_shapes, topk, num=2):
        level0 = level_start_index[0].item()
        level1 = level_start_index[1].item()
        level2 = level_start_index[2].item()
        level3 = level_start_index[3].item()
    
        l1_h, l1_w = spatial_shapes[0][0], spatial_shapes[0][1]
        l2_h, l2_w = spatial_shapes[1][0], spatial_shapes[1][1]
        l3_h, l3_w = spatial_shapes[2][0], spatial_shapes[2][1]
        l4_h, l4_w = spatial_shapes[3][0], spatial_shapes[3][1]

        N, _, d = output_memory.shape

        fea1 = enc_outputs_class_unselected[:, level0:level1, :]
        fea2 = enc_outputs_class_unselected[:, level1:level2, :]
        fea3 = enc_outputs_class_unselected[:, level2:level3, :]
        fea4 = enc_outputs_class_unselected[:, level3:, :]

        l1 = 400; l2 = 300; l3 = 140; l4 = 60
        num_candidate = 2
        val1, indice1 = torch.topk(fea1.max(-1)[0], l1, dim=1) 
        val2, indice2 = torch.topk(fea2.max(-1)[0], l2, dim=1)
        val3, indice3 = torch.topk(fea3.max(-1)[0], l3, dim=1)
        val4, indice4 = torch.topk(fea4.max(-1)[0], l4, dim=1)

        indice2 = indice2 + level1
        indice3 = indice3 + level2
        indice4 = indice4 + level3

        out_indice = torch.cat([indice1, indice2, indice3, indice4], dim=1)
        out_value = torch.cat([val1, val2, val3, val4], dim=1)

        return out_indice, out_value


    def get_distance_metric(self, fea_indice, confidence, h, w, threshold):
        """
        索引，单个值，所有向量
        """
        
        fea_indice = fea_indice
        remain = fea_indice % w
        height = torch.div(fea_indice, w, rounding_mode='floor')

        h_1 = height
        h_2 = height.T

        w_1 = remain
        w_2 = remain.T

        dis = torch.square(h_1 - h_2) + torch.square(w_1 - w_2)
        dis = torch.sqrt(dis)   # A  乘以阈值变差了

        a = dis <= threshold
        dis = a * dis

        dis = dis + torch.eye(dis.shape[0]).cuda()

        sum_dis = torch.sum(dis, dim=1)
        c = dis > 0
        num_dis = torch.sum(c, dim=1)
        mean = sum_dis / num_dis

        std = torch.sqrt(1.0/num_dis * torch.sum(c*(dis-mean)**2, dim=1))

        cv = std / mean

        # miu = torch.mean(dis, dim=1)
        # sig = torch.std(dis, dim=1)
        # cv = sig / miu

        # return torch.sigmoid(mean)
        return cv

    
    def feature_multi_scale(self, feature, level_start_index, spatial_shapes):
        """
        输入为四层的特征，输出为不同层融合其他层后的特征，为多尺度特征，分别对应不同尺度的目标
        """
        level0 = level_start_index[0].item()
        level1 = level_start_index[1].item()
        level2 = level_start_index[2].item()
        level3 = level_start_index[3].item()

        l1_h, l1_w = spatial_shapes[0][0], spatial_shapes[0][1]
        l2_h, l2_w = spatial_shapes[1][0], spatial_shapes[1][1]
        l3_h, l3_w = spatial_shapes[2][0], spatial_shapes[2][1]
        l4_h, l4_w = spatial_shapes[3][0], spatial_shapes[3][1]

        N, _, d = feature.shape

        fea1 = feature[:, level0:level1, :].view(N, l1_h, l1_w, d).permute(0, 3, 1, 2)
        fea2 = feature[:, level1:level2, :].view(N, l2_h, l2_w, d).permute(0, 3, 1, 2)
        fea3 = feature[:, level2:level3, :].view(N, l3_h, l3_w, d).permute(0, 3, 1, 2)
        fea4 = feature[:, level3:, :].view(N, l4_h, l4_w, d).permute(0, 3, 1, 2)

        fea1 = fea1.mean(dim=[2,3]).flatten(1)
        fea2 = fea2.mean(dim=[2,3]).flatten(1)
        fea3 = fea3.mean(dim=[2,3]).flatten(1)
        fea4 = fea4.mean(dim=[2,3]).flatten(1)

        # feas = fea1 + fea2 + fea3 + fea4
        out = self.class_fuse(torch.cat([fea1, fea2, fea3, fea4], dim=1))

        return out

    def feature_multi_scale_enc(self, feature):
        """
        输入为四层的特征，输出为不同层融合其他层后的特征，为多尺度特征，分别对应不同尺度的目标
        """
        # fea1 = feature[0].mean(dim=[2,3]).flatten(1)
        # fea2 = feature[1].mean(dim=[2,3]).flatten(1)
        # fea3 = feature[2].mean(dim=[2,3]).flatten(1)
        fea4 = feature[3].mean(dim=[2,3]).flatten(1)

        # feas = fea1 + fea2 + fea3 + fea4
        # out = self.class_fuse_enc(torch.cat([fea1, fea2, fea3, fea4], dim=1))
        out = self.class_fuse_enc(fea4)

        return out


    def conv_(self, feature, level_start_index, spatial_shapes):
        level0 = level_start_index[0].item()
        level1 = level_start_index[1].item()
        level2 = level_start_index[2].item()
        level3 = level_start_index[3].item()

        l1_h, l1_w = spatial_shapes[0][0], spatial_shapes[0][1]
        l2_h, l2_w = spatial_shapes[1][0], spatial_shapes[1][1]
        l3_h, l3_w = spatial_shapes[2][0], spatial_shapes[2][1]
        l4_h, l4_w = spatial_shapes[3][0], spatial_shapes[3][1]

        N, _, d = feature.shape

        fea1 = feature[:, level0:level1, :].view(N, l1_h, l1_w, d).permute(0, 3, 1, 2)
        fea2 = feature[:, level1:level2, :].view(N, l2_h, l2_w, d).permute(0, 3, 1, 2)
        fea3 = feature[:, level2:level3, :].view(N, l3_h, l3_w, d).permute(0, 3, 1, 2)
        fea4 = feature[:, level3:, :].view(N, l4_h, l4_w, d).permute(0, 3, 1, 2)

        fea1 = self.conv1(fea1).view(N, d, l1_h * l1_w).permute(0, 2, 1)
        fea2 = self.conv2(fea2).view(N, d, l2_h * l2_w).permute(0, 2, 1)
        fea3 = self.conv3(fea3).view(N, d, l3_h * l3_w).permute(0, 2, 1)
        fea4 = self.conv4(fea4).view(N, d, l4_h * l4_w).permute(0, 2, 1)

        # fea1 = self.conv1(fea1).view(N, d, l1_h * l1_w).permute(0, 2, 1)
        # fea2 = self.conv1(fea2).view(N, d, l2_h * l2_w).permute(0, 2, 1)
        # fea3 = self.conv1(fea3).view(N, d, l3_h * l3_w).permute(0, 2, 1)
        # fea4 = self.conv1(fea4).view(N, d, l4_h * l4_w).permute(0, 2, 1)

        fea = torch.cat([fea1, fea2, fea3, fea4], dim=1)

        if feature.shape[1] != fea.shape[1]:
            print()

        return fea


    # def nms_query(self, enc_outputs_coord_unselected, enc_outputs_class_unselected):
    #     # NMS for selected query
    #     box_xyxy = box_ops.box_cxcywh_to_xyxy(enc_outputs_coord_unselected.sigmoid())
    #     class_score = enc_outputs_class_unselected.sigmoid().max(-1)[0]
    #     class_ = torch.argmax(enc_outputs_class_unselected, dim=2)
    #     # x = [nms(b, s, iou_threshold=0.7) for b,s in zip(box_xyxy, class_score)]
    #     x = [batched_nms(b, s, id, iou_threshold=0.3) for b, s, id in zip(box_xyxy, class_score, class_)]
    #     # x2 = [nms(b, s, iou_threshold=0.6) for b,s in zip(box_xyxy, class_score)]
        
    #     max_num = 0; min_num = 10000; cur_num = []
    #     for i in range(len(x)):
    #         s = x[i].shape[0]
    #         cur_num.append(s)
    #         if max_num < s:
    #             max_num = s

    #     if max_num < self.num_queries:
    #         max_num = self.num_queries
        
    #     print("nms data:", cur_num)

    #     num_supp = []
    #     max_scores = enc_outputs_class_unselected.max(-1)[0]
    #     topk_proposals = None
    #     for i in range(len(cur_num)):
    #         supp = max_num - cur_num[i]
    #         num_supp.append(supp)
    #         if supp > 0:
    #             value_, topk_pro = torch.topk(max_scores[i:i+1, :], supp, dim=1) # bs, nq
    #         else:
    #             topk_pro = None
    #         if topk_proposals is None:
    #             if topk_pro is not None:
    #                 topk_proposals = torch.cat([x[i].unsqueeze(0), topk_pro], dim=1)
    #             else:
    #                 topk_proposals = x[i].unsqueeze(0)
    #         else:
    #             if topk_pro is not None:
    #                 cur_ = torch.cat([x[i].unsqueeze(0), topk_pro], dim=1)
    #                 topk_proposals = torch.cat([topk_proposals, cur_], dim=0)
    #             else:
    #                 cur_ = x[i].unsqueeze(0)
    #                 topk_proposals = torch.cat([topk_proposals, cur_], dim=0)
        
    #     return topk_proposals

    def nms_query_low_score(self, enc_outputs_coord_unselected, enc_outputs_class_unselected, spatial_shapes):
        # NMS for selected query
        box_xyxy = box_ops.box_cxcywh_to_xyxy(enc_outputs_coord_unselected.sigmoid())
        h, w = spatial_shapes[0][0]*4, spatial_shapes[0][1]*4
        scale_fct = torch.stack([w, h, w, h], dim=0)
        scale_fct = scale_fct.unsqueeze(0).unsqueeze(0)
        box_xyxy = box_xyxy * scale_fct
        class_score = enc_outputs_class_unselected.sigmoid().max(-1)[0]
        x = [nms(b, s, iou_threshold=self.nms_th_free) for b,s in zip(box_xyxy, class_score)]

        # class_ = torch.argmax(enc_outputs_class_unselected, dim=2)
        # x = [batched_nms(b, s, id, iou_threshold=self.nms_th_free) for b, s, id in zip(box_xyxy, class_score, class_)]
        
        max_num = 0; min_num = 10000; cur_num = []
        for i in range(len(x)):
            s = x[i].shape[0]
            cur_num.append(s)
            if max_num < s:
                max_num = s

        # if max_num < self.num_queries:
        #     max_num = self.num_queries
        
        print("nms data:", cur_num, "max:", max_num)

        num_supp = []
        max_scores = enc_outputs_class_unselected.max(-1)[0].sigmoid()
        topk_proposals = None
        for i in range(len(cur_num)):
            supp = max_num - cur_num[i]
            num_supp.append(supp)
            if supp > 0:
                value_, topk_pro = torch.topk(-max_scores[i:i+1, :], supp, dim=1) # bs, nq
            else:
                topk_pro = None
            if topk_proposals is None:
                if topk_pro is not None:
                    topk_proposals = torch.cat([x[i].unsqueeze(0), topk_pro], dim=1)
                else:
                    topk_proposals = x[i].unsqueeze(0)
            else:
                if topk_pro is not None:
                    cur_ = torch.cat([x[i].unsqueeze(0), topk_pro], dim=1)
                    topk_proposals = torch.cat([topk_proposals, cur_], dim=0)
                else:
                    cur_ = x[i].unsqueeze(0)
                    topk_proposals = torch.cat([topk_proposals, cur_], dim=0)
        
        return topk_proposals

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


    def binary_selected(self, binary, enc_outputs_class_unselected):
        # class_pred = enc_outputs_class_unselected.max(-1)[0].sigmoid()
        # binary = class_pred.unsqueeze(-1) * binary    # 0.005
        print("enc binary max:", binary.max().item(), "  min:", binary.min().item(), "mean: ", binary.mean().item(),\
              " std:", binary.std().item())
        # self.save_his_img(binary)

        B, N, C = binary.shape
        max_ = -1; number_list = []
        selected = []
        for i in range(B):
            tp = torch.where(binary[i, :, :].squeeze(-1) >= 0.02)
            
            # mean_ = binary[i, :, :].mean(); 
            # # std_ = binary[i, :, :].std()
            # tp = torch.where(binary[i, :, :].squeeze(-1) >= mean_)
            # print("pro:", mean_.item())

            # tp = torch.where(binary[i, :, :].squeeze(-1) >= 0.1)

            # tp = torch.where(binary[i, :, :].squeeze(-1) >= 0.001)

            # tp = torch.where(class_pred[i, :] >= 0.01)
            number_list.append(tp[0].shape[0])
            if max_ < tp[0].shape[0]:
                max_ = tp[0].shape[0]
            selected.append(tp[0])

        # if max_ < 100:    # 设置最小query
        #     max_ = 100

        final_max = 0
        if max_ == 0:
            final_max = 100
        else:
            final_max = max_
        print("query num:", max_)

        # if max_ == 15070:
        #     print()

        max_scores = enc_outputs_class_unselected.max(-1)[0].sigmoid()
        topk_proposals = None
        for i in range(B):
            supp = final_max - selected[i].shape[0]
            if supp > 0:
                if max_ == 0:
                    value_, topk_pro = torch.topk(max_scores[i:i+1, :], supp, dim=1) # bs, nq
                else:
                    value_, topk_pro = torch.topk(-max_scores[i:i+1, :], supp, dim=1) # bs, nq
            else:
                topk_pro = None
            if topk_proposals is None:
                if topk_pro is not None:
                    topk_proposals = torch.cat([selected[i].unsqueeze(0), topk_pro], dim=1)
                else:
                    topk_proposals = selected[i].unsqueeze(0)
            else:
                if topk_pro is not None:
                    cur_ = torch.cat([selected[i].unsqueeze(0), topk_pro], dim=1)
                    topk_proposals = torch.cat([topk_proposals, cur_], dim=0)
                else:
                    cur_ = selected[i].unsqueeze(0)
                    topk_proposals = torch.cat([topk_proposals, cur_], dim=0)

        return topk_proposals 

    def forward(self, srcs, masks, refpoint_embed, pos_embeds, tgt, attn_mask=None, max_boxes=0):
        """
        Input:
            - srcs: List of multi features [bs, ci, hi, wi]
            - masks: List of multi masks [bs, hi, wi]
            - refpoint_embed: [bs, num_dn, 4]. None in infer
            - pos_embeds: List of multi pos embeds [bs, ci, hi, wi]
            - tgt: [bs, num_dn, d_model]. None in infer
            
        """
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)                # bs, hw, c
            mask = mask.flatten(1)                              # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)    # bs, hw, c
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)    # bs, \sum{hxw}, c 
        mask_flatten = torch.cat(mask_flatten, 1)   # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) # bs, \sum{hxw}, c 
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # two stage
        enc_topk_proposals = enc_refpoint_embed = None

        #########################################################
        # Begin Encoder
        #########################################################
        memory, enc_intermediate_output, enc_intermediate_refpoints = self.encoder(
                src_flatten, 
                pos=lvl_pos_embed_flatten, 
                level_start_index=level_start_index, 
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios,
                key_padding_mask=mask_flatten,
                ref_token_index=enc_topk_proposals, # bs, nq 
                ref_token_coord=enc_refpoint_embed, # bs, nq, 4
                )
        #########################################################
        # End Encoder
        # - memory: bs, \sum{hw}, c
        # - mask_flatten: bs, \sum{hw}
        # - lvl_pos_embed_flatten: bs, \sum{hw}, c
        # - enc_intermediate_output: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        # - enc_intermediate_refpoints: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        #########################################################

        if self.two_stage_type =='standard':
            if self.two_stage_learn_wh:
                input_hw = self.two_stage_wh_embedding.weight[0]
            else:
                input_hw = None
            output_memory, output_proposals = gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes, input_hw)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            
            if self.two_stage_pat_embed > 0:
                bs, nhw, _ = output_memory.shape
                # output_memory: bs, n, 256; self.pat_embed_for_2stage: k, 256
                output_memory = output_memory.repeat(1, self.two_stage_pat_embed, 1)
                _pats = self.pat_embed_for_2stage.repeat_interleave(nhw, 0) 
                output_memory = output_memory + _pats
                output_proposals = output_proposals.repeat(1, self.two_stage_pat_embed, 1)

            if self.two_stage_add_query_num > 0:
                assert refpoint_embed is not None
                output_memory = torch.cat((output_memory, tgt), dim=1)
                output_proposals = torch.cat((output_proposals, refpoint_embed), dim=1)

            # output_memory_binary = output_memory * binary

            enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
            # 对输入特征进行权重处理
            # binary_fea = self.conv_(output_memory.detach(), level_start_index, spatial_shapes)
            if self.enc_exsa is not None:
                # binary_fea = self.enc_norm(output_memory + self.enc_exsa(output_memory))
                binary_fea = output_memory + self.enc_exsa(output_memory).sigmoid() * output_memory
            else:
                binary_fea = output_memory

            binary = self.enc_binary_class(binary_fea).sigmoid()

            if self.conditional_binary_enc is not None:                
                # binary_ = (binary * 100).int()
                # weight_ = self.conditional_binary_enc(binary_.view(-1))
                # b, n, _ = binary.shape
                # enc_outputs_coord_unselected = self.enc_out_bbox_embed(output_memory+weight_.view(b,n,-1) * binary) + output_proposals # (bs, \sum{hw}, 4) unsigmoid
                enc_outputs_coord_unselected = self.enc_out_bbox_embed(output_memory * (binary + 1.0)) + output_proposals # (bs, \sum{hw}, 4) unsigmoid
            else:
                enc_outputs_coord_unselected = self.enc_out_bbox_embed(binary_fea) + output_proposals # (bs, \sum{hw}, 4) unsigmoid
            # enc_outputs_binary = self.enc_binary_class(self.enc_mlp(output_memory))
            # score = enc_outputs_binary.sigmoid()
            # indices_ = score[:, :, 0] > score.mean() + score.std() * 3
            # print("seleced num:", indices_.sum())

            if max_boxes > 0:
                topk = max_boxes * 2
            else:
                topk = self.num_queries

            # ours query with position
            # topk_proposals, value_sort = self.select_query_with_position_only_firstlayer(level_start_index, enc_outputs_class_unselected, spatial_shapes, topk, num=2)
            # topk_proposals, value_sort = self.select_query_with_position_3(level_start_index, enc_outputs_class_unselected, spatial_shapes, topk, num=2)


            # origin 方法
            # value_, topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1) # bs, nq
            
            # 选择5 points方法
            # topk_proposals = torch.topk(torch.mean(torch.topk(enc_outputs_class_unselected, 5, dim=2)[0], dim=2), topk, dim=1)[1] # bs, nq

            # top k from different scale layers
            # feas = enc_outputs_class_unselected.max(-1)[0]
            # size = feas.shape[1]; 
            # sz4 = int((size - level_start_index[3].item()) * 4/ size * topk)
            # sz2 = int((level_start_index[2].item() - level_start_index[1].item()) * 2/ size * topk)
            # sz3 = int((level_start_index[3].item() - level_start_index[2].item()) * 3 / size * topk)

            # f1 = torch.topk(feas[:, level_start_index[0].item():level_start_index[1].item()], topk-sz4-sz3-sz2, dim=1)[1]    # 第一层
            # f2 = torch.topk(feas[:, level_start_index[1].item():level_start_index[2].item()], sz2, dim=1)[1]    # 第一层
            # f3 = torch.topk(feas[:, level_start_index[2].item():level_start_index[3].item()], sz3, dim=1)[1]    # 第一层
            # f4 = torch.topk(feas[:, level_start_index[3].item():], sz4, dim=1)[1]    # 第一层
            # topk_proposals = torch.cat([f1, f2, f3, f4], dim=1)

            # import time
            # name = time.time()
            # self.save_quries_distribution(level_start_index, enc_outputs_class_unselected, 900, spatial_shapes, "origin_" + str(name), value_)
            
            # value_sort, topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], enc_outputs_class_unselected.shape[1], dim=1) # bs, nq
            # self.save_quries_distribution_pos(level_start_index, topk_proposals, value_sort, enc_outputs_class_unselected.shape[1], spatial_shapes, "o_ours_" + str(name))

            if not self.gqs_use:
                max_scores = enc_outputs_class_unselected.max(-1)[0]
                # value_, topk_proposals = torch.topk(max_scores, topk, dim=1) # bs, nq

                if self.nms_decoder:
                    topk_proposals = self.nms_query_low_score(enc_outputs_coord_unselected, enc_outputs_class_unselected, spatial_shapes)
            
                # binary筛选
                if self.binary:
                    topk_proposals = self.binary_selected(binary, enc_outputs_class_unselected)


                # # 取所有的query，进行训练
                # b, n = enc_outputs_coord_unselected.shape[0], enc_outputs_coord_unselected.shape[1]
                # value_, topk_proposals = torch.topk(max_scores, n, dim=1) # bs, nq

                # gather boxes
                refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)) # unsigmoid
                refpoint_embed_ = refpoint_embed_undetach.detach()
                init_box_proposal = torch.gather(output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).sigmoid() # sigmoid

                # gather tgt
                # tgt_undetach = torch.gather(output_memory_binary, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))
                tgt_undetach_class = torch.gather(enc_outputs_class_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.num_class))
                tgt_undetach_binary = torch.gather(self.enc_binary_class(binary_fea), 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 1))
            else:
                value_, topk_proposals_first = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1) # bs, nq
            
                # 前两个为选择出的900个query，后两个为前1800个query，用于计算loss
                topk_proposals, value_sort = self.select_query_with_position_4(level_start_index, output_memory, enc_outputs_class_unselected, spatial_shapes, topk, num=2)

                # refpoint_embed_undetach_ori = torch.gather(enc_outputs_coord_unselected, 1, topk_proposals_first.unsqueeze(-1).repeat(1, 1, 4)) # unsigmoid

                # topk_proposals = torch.cat([topk_proposals_first[:, :450], topk_proposals[:, :450]], dim=1)  # 各取一半

                # 采用更新后的向量，初始化点
                # refpoint_embed_undetach = self.enc_out_bbox_embed(data_topk_v) + torch.gather(output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
                refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)) # unsigmoid

                # refpoint_embed_undetach2 = torch.gather(enc_outputs_coord_unselected, 1, topk_proposals_2.unsqueeze(-1).repeat(1, 1, 4)) # unsigmoid
                refpoint_embed_ = refpoint_embed_undetach.detach()
                init_box_proposal = torch.gather(output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).sigmoid() # sigmoid

            # gather data for src_flatten
            # src_flat_pro = self.backbone_process(src_flatten.unsqueeze(3).permute(0, 2,1,3))  # N Q C 1 → N C Q 1
            # src_flat_pro = self.backbone_norm(src_flat_pro.permute(0, 2, 1, 3).squeeze(3))
            # src_flat_pro = nn.GELU()(src_flat_pro)
            # # src_flatten = torch.sigmoid(memory) * src_flatten
            # backbone_coord = self.backbone_out_bbox_embed(src_flat_pro) + output_proposals
            # src_flatten_return = torch.gather(src_flat_pro, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 256))
            # src_flatten_coord = torch.gather(backbone_coord, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            src_flatten_return = None
            src_flatten_coord = None

            # memory = self.backbone_fuse(torch.cat([memory, src_flat_pro], dim=2))

            # gather tgt
            if self.embed_init_tgt:
                tgt_undetach = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))
                tgt_undetach_bias_4 = None
            else:
                tgt_undetach = torch.gather(binary_fea, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))
                
                # 分类结果的利用
                # tgt_undetach = tgt_undetach*(value_.sigmoid().unsqueeze(dim=2) + 1.0)

                # tgt_undetach_bias = self.head_tgt_bias(tgt_undetach) + tgt_undetach
                # tgt_undetach_bias = self.bias_norm(tgt_undetach_bias)

                # tgt_undetach_bias = tgt_undetach_bias * (1 + nn.functional.softmax(tgt_undetach_bias, dim=1))
                # tgt_undetach_bias = self.activation_gelu(self.bias_norm(tgt_undetach))
                tgt_undetach_bias_4 = None



            # import time
            # name = time.time()
            # # self.save_quries_distribution(level_start_index, enc_outputs_class_unselected, 900, spatial_shapes, "origin_" + str(name), value_)
            # self.save_quries_distribution_pos(level_start_index, topk_proposals, value_sort, enc_outputs_class_unselected.shape[1], spatial_shapes, "o_ours_" + str(name))

            if self.embed_init_tgt:
                tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1) # nq, bs, d_model
                tgt_undetach_bias_4 = None
            else:
                tgt_ = tgt_undetach.detach()
                # tgt_ = tgt_undetach
                # tgt_ = tgt_undetach_bias.detach()
                # tgt_ = tgt_undetach_bias


            if refpoint_embed is not None:
                refpoint_embed=torch.cat([refpoint_embed,refpoint_embed_],dim=1)

                tgt_undetach_class= torch.cat([self.enc_out_class_embed(tgt), tgt_undetach_class], dim=1)
                tgt_undetach_binary = torch.cat([self.enc_binary_class(tgt), tgt_undetach_binary], dim=1)

                tgt=torch.cat([tgt,tgt_],dim=1)
            else:
                refpoint_embed,tgt=refpoint_embed_,tgt_

        elif self.two_stage_type == 'no':
            tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1) # nq, bs, d_model
            refpoint_embed_ = self.refpoint_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1) # nq, bs, 4

            if refpoint_embed is not None:
                refpoint_embed=torch.cat([refpoint_embed,refpoint_embed_],dim=1)
                tgt=torch.cat([tgt,tgt_],dim=1)
            else:
                refpoint_embed,tgt=refpoint_embed_,tgt_

            if self.num_patterns > 0:
                tgt_embed = tgt.repeat(1, self.num_patterns, 1)
                refpoint_embed = refpoint_embed.repeat(1, self.num_patterns, 1)
                tgt_pat = self.patterns.weight[None, :, :].repeat_interleave(self.num_queries, 1) # 1, n_q*n_pat, d_model
                tgt = tgt_embed + tgt_pat

            init_box_proposal = refpoint_embed_.sigmoid()

        else:
            raise NotImplementedError("unknown two_stage_type {}".format(self.two_stage_type))
        #########################################################
        # End preparing tgt
        # - tgt: bs, NQ, d_model
        # - refpoint_embed(unsigmoid): bs, NQ, d_model 
        ######################################################### 

        if self.cross_coding:
            memory = self.fusion(src_flatten, memory, level_start_index, spatial_shapes)
        
            backbone_coord = self.backbone_out_bbox_embed(memory) + output_proposals
            src_flatten_return = torch.gather(memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 256))
            src_flatten_coord = torch.gather(backbone_coord, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))

        # memory, enc_intermediate_output, enc_intermediate_refpoints = self.encoder_fusion(
        #         memory,
        #         src_flatten, 
        #         pos=lvl_pos_embed_flatten, 
        #         level_start_index=level_start_index,
        #         spatial_shapes=spatial_shapes,
        #         valid_ratios=valid_ratios,
        #         key_padding_mask=mask_flatten,
        #         ref_token_index=enc_topk_proposals, # bs, nq 
        #         ref_token_coord=enc_refpoint_embed, # bs, nq, 4
        #         )


        #########################################################
        # Begin Decoder
        #########################################################
        hs, references, nms_selected, dn_numbers = self.decoder(
                tgt=tgt.transpose(0, 1), 
                memory=memory.transpose(0, 1), 
                memory_key_padding_mask=mask_flatten, 
                pos=lvl_pos_embed_flatten.transpose(0, 1),
                refpoints_unsigmoid=refpoint_embed.transpose(0, 1), 
                level_start_index=level_start_index, 
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios,tgt_mask=attn_mask, topk_proposals=topk_proposals,
                tgt_undetach_class=tgt_undetach_class,
                tgt_undetach_binary=tgt_undetach_binary,)
        #########################################################
        # End Decoder
        # hs: n_dec, bs, nq, d_model
        # references: n_dec+1, bs, nq, query_dim
        #########################################################

        #########################################################
        # Begin postprocess
        #########################################################     
        if self.two_stage_type == 'standard':
            if self.two_stage_keep_all_tokens:
                hs_enc = output_memory.unsqueeze(0)
                ref_enc = enc_outputs_coord_unselected.sigmoid().unsqueeze(0)
                init_box_proposal = output_proposals
                gqs_enc = None
                ref_gqs_enc = None
            else:
                if not self.gqs_use:
                    hs_enc = tgt_undetach.unsqueeze(0)
                    ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
                    gqs_enc = None
                    ref_gqs_enc = None
                elif self.gqs_use:
                    # hs_enc = tgt_undetach.unsqueeze(0)
                    # ref_enc = refpoint_embed_undetach_ori.sigmoid().unsqueeze(0)
                    # gqs_enc = data_topk_v.unsqueeze(0)    # 
                    # ref_gqs_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)

                    hs_enc = tgt_undetach.unsqueeze(0)    # 
                    ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
                    gqs_enc = tgt_undetach.unsqueeze(0)    # 
                    ref_gqs_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
        else:
            hs_enc = ref_enc = None
        #########################################################
        # End postprocess
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or (n_enc, bs, nq, d_model) or None
        # ref_enc: (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or (n_enc, bs, nq, d_model) or None
        #########################################################        

        # 用于多分类loss
        if self.gcs_transformer_enc:
            gcs_transformer_out = self.feature_multi_scale(output_memory, level_start_index, spatial_shapes)
        else:
            gcs_transformer_out = None
        
        if self.gcs_enc:
            gcs_back_out = self.feature_multi_scale_enc(srcs)
        else:
            gcs_back_out = None

        return hs, references, nms_selected, dn_numbers, hs_enc, gcs_transformer_out, gcs_back_out, gqs_enc, ref_gqs_enc, ref_enc, init_box_proposal, tgt_undetach_bias_4, src_flatten_return, src_flatten_coord, level_start_index, spatial_shapes, memory, topk_proposals
        # hs: (n_dec, bs, nq, d_model)
        # references: sigmoid coordinates. (n_dec+1, bs, bq, 4)
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or None
        # ref_enc: sigmoid coordinates. \
        #           (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or None


class TransformerEncoder(nn.Module):

    def __init__(self, 
        encoder_layer, num_layers, norm=None, d_model=256, 
        num_queries=300,
        deformable_encoder=False, 
        enc_layer_share=False, enc_layer_dropout_prob=None,                  
        two_stage_type='no',  # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
    ):
        super().__init__()
        # prepare layers
        if num_layers > 0:
            self.layers = _get_clones(encoder_layer, num_layers, layer_share=enc_layer_share)
        else:
            self.layers = []
            del encoder_layer

        self.query_scale = None
        self.num_queries = num_queries
        self.deformable_encoder = deformable_encoder
        self.num_layers = num_layers
        self.norm = norm
        self.d_model = d_model

        self.enc_layer_dropout_prob = enc_layer_dropout_prob
        if enc_layer_dropout_prob is not None:
            assert isinstance(enc_layer_dropout_prob, list)
            assert len(enc_layer_dropout_prob) == num_layers
            for i in enc_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self.two_stage_type = two_stage_type
        if two_stage_type in ['enceachlayer', 'enclayer1']:
            _proj_layer = nn.Linear(d_model, d_model)
            _norm_layer = nn.LayerNorm(d_model)
            if two_stage_type == 'enclayer1':
                self.enc_norm = nn.ModuleList([_norm_layer])
                self.enc_proj = nn.ModuleList([_proj_layer])
            else:
                self.enc_norm = nn.ModuleList([copy.deepcopy(_norm_layer) for i in range(num_layers - 1) ])
                self.enc_proj = nn.ModuleList([copy.deepcopy(_proj_layer) for i in range(num_layers - 1) ]) 

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, 
            src: Tensor, 
            pos: Tensor, 
            spatial_shapes: Tensor, 
            level_start_index: Tensor, 
            valid_ratios: Tensor, 
            key_padding_mask: Tensor,
            ref_token_index: Optional[Tensor]=None,
            ref_token_coord: Optional[Tensor]=None 
            ):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]

            - ref_token_index: bs, nq
            - ref_token_coord: bs, nq, 4
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus: 
            - output: [bs, sum(hi*wi), 256]
        """
        if self.two_stage_type in ['no', 'standard', 'enceachlayer', 'enclayer1']:
            assert ref_token_index is None

        output = src
        # preparation and reshape
        if self.num_layers > 0:
            if self.deformable_encoder:
                reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)

        intermediate_output = []
        intermediate_ref = []
        if ref_token_index is not None:
            out_i = torch.gather(output, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, self.d_model))
            intermediate_output.append(out_i)
            intermediate_ref.append(ref_token_coord)

        # main process
        for layer_id, layer in enumerate(self.layers):
            # main process
            dropflag = False
            if self.enc_layer_dropout_prob is not None:
                prob = random.random()
                if prob < self.enc_layer_dropout_prob[layer_id]:
                    dropflag = True
            
            if not dropflag:
                if self.deformable_encoder:
                    output = layer(src=output, pos=pos, reference_points=reference_points, spatial_shapes=spatial_shapes, level_start_index=level_start_index, key_padding_mask=key_padding_mask)  
                else:
                    output = layer(src=output.transpose(0, 1), pos=pos.transpose(0, 1), key_padding_mask=key_padding_mask).transpose(0, 1)        

            if ((layer_id == 0 and self.two_stage_type in ['enceachlayer', 'enclayer1']) \
                or (self.two_stage_type == 'enceachlayer')) \
                    and (layer_id != self.num_layers - 1):
                output_memory, output_proposals = gen_encoder_output_proposals(output, key_padding_mask, spatial_shapes)
                output_memory = self.enc_norm[layer_id](self.enc_proj[layer_id](output_memory))
                
                # gather boxes
                topk = self.num_queries
                enc_outputs_class = self.class_embed[layer_id](output_memory)
                ref_token_index = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1] # bs, nq
                ref_token_coord = torch.gather(output_proposals, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, 4))

                output = output_memory

            # aux loss
            if (layer_id != self.num_layers - 1) and ref_token_index is not None:
                out_i = torch.gather(output, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, self.d_model))
                intermediate_output.append(out_i)
                intermediate_ref.append(ref_token_coord)

        if self.norm is not None:
            output = self.norm(output)

        if ref_token_index is not None:
            intermediate_output = torch.stack(intermediate_output) # n_enc/n_enc-1, bs, \sum{hw}, d_model
            intermediate_ref = torch.stack(intermediate_ref)
        else:
            intermediate_output = intermediate_ref = None

        return output, intermediate_output, intermediate_ref

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, 
                    return_intermediate=False, 
                    d_model=256, query_dim=4, 
                    modulate_hw_attn=False,
                    num_feature_levels=1,
                    deformable_decoder=False,
                    decoder_query_perturber=None,
                    dec_layer_number=None, # number of queries each layer in decoder
                    rm_dec_query_scale=False,
                    dec_layer_share=False,
                    dec_layer_dropout_prob=None,
                    use_detached_boxes_dec_out=False,
                    nms_decoder=0,
                    binary=False,
                    ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers, layer_share=dec_layer_share)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.nms_decoder = nms_decoder
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out

        
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None

        if rm_dec_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.bbox_embed1 = None
        self.class_embed1 = None
        self.binary = binary
        self.mlp = None    # 为tgt做条件处理
        self.binary_embed = None
        self.conditional_binary = None
        # self.bbox_embed2 = None
        # self.class_embed2 = None

        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder

        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:
            self.ref_anchor_head = None

        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None

        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_layers
            
        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self.rm_detach = None


    def pred_in_4_layers(self, layer1, layer2, layer_id, fea, level_start_index, topk_proposals):
        """
        分层预测，采用不同的参数
        """
        level0 = level_start_index[0].item()
        level1 = level_start_index[1].item()
        level2 = level_start_index[2].item()
        level3 = level_start_index[3].item()

        dn_num = fea.shape[0] - topk_proposals.shape[1]
        if dn_num > 0:
            fea_dn = fea[:dn_num, :, :]
            fea_q = fea[dn_num:, :, :]
        else:
            fea_q = fea

        id_sort_value, sort_indice = torch.sort(topk_proposals, 1)
        sort_fea = torch.gather(fea_q, 0, sort_indice.permute(1, 0).unsqueeze(dim=2).repeat(1, 1, 
                                                    fea_q.shape[2]))
        v_, s_id = torch.sort(sort_indice, 1)

        # for layers
        idx0 = torch.where(id_sort_value > level1)[1][0]
        fea1 = layer1[layer_id](sort_fea[0:idx0, :, :])
        
        # idx1 = torch.where(id_sort_value >= level1)[1][0]
        fea2 = layer2[layer_id](sort_fea[idx0:, :, :])
        
        # idx2 = torch.where(id_sort_value > level3)[1][0]
        # fea3 = layer[2].cuda()[layer_id](sort_fea[idx1:idx2, :, :])
        
        # fea4 = layer[3].cuda()[layer_id](sort_fea[idx2:, :, :])

        new_fea = torch.cat([fea1, fea2], dim=0)
        new_fea = torch.gather(new_fea, 0, s_id.permute(1, 0).unsqueeze(dim=2).repeat(1, 1, 
                                                    new_fea.shape[2]))
        if dn_num > 0:
            new_fea = torch.cat([0.5*(layer1[layer_id](fea_dn) + layer2[layer_id](fea_dn)), new_fea], dim=0)
        
        return new_fea

    def nms_query_low_score(self, iou_threshold, enc_outputs_coord_unselected, enc_outputs_class_unselected, spatial_shapes):
        # NMS for selected query
        box_xyxy = box_ops.box_cxcywh_to_xyxy(enc_outputs_coord_unselected)
        h, w = spatial_shapes[0][0]*4, spatial_shapes[0][1]*4
        scale_fct = torch.stack([w, h, w, h], dim=0)
        scale_fct = scale_fct.unsqueeze(0).unsqueeze(0)
        box_xyxy = box_xyxy * scale_fct

        class_score = enc_outputs_class_unselected.sigmoid().max(-1)[0]
        class_ = torch.argmax(enc_outputs_class_unselected, dim=2)
        x = [nms(b, s, iou_threshold=iou_threshold) for b,s in zip(box_xyxy, class_score)]
        # x = [batched_nms(b, s, id, iou_threshold=0.95) for b, s, id in zip(box_xyxy, class_score, class_)]
        # x = [nms(b, s, iou_threshold=self.nms_th_free) for b,s in zip(box_xyxy, class_score)]
        
        max_num = 0; min_num = 10000; cur_num = []
        for i in range(len(x)):
            s = x[i].shape[0]
            cur_num.append(s)
            if max_num < s:
                max_num = s

        # if max_num < self.num_queries:
        #     max_num = self.num_queries
        
        # print("nms data:", cur_num, "max:", max_num)

        num_supp = []
        max_scores = enc_outputs_class_unselected.max(-1)[0]
        topk_proposals = None
        for i in range(len(cur_num)):
            supp = max_num - cur_num[i]
            num_supp.append(supp)
            if supp > 0:
                value_, topk_pro = torch.topk(-max_scores[i:i+1, :].sigmoid(), supp, dim=1) # bs, nq
            else:
                topk_pro = None
            if topk_proposals is None:
                if topk_pro is not None:
                    topk_proposals = torch.cat([x[i].unsqueeze(0), topk_pro], dim=1)
                else:
                    topk_proposals = x[i].unsqueeze(0)
            else:
                if topk_pro is not None:
                    cur_ = torch.cat([x[i].unsqueeze(0), topk_pro], dim=1)
                    topk_proposals = torch.cat([topk_proposals, cur_], dim=0)
                else:
                    cur_ = x[i].unsqueeze(0)
                    topk_proposals = torch.cat([topk_proposals, cur_], dim=0)
        
        return topk_proposals


    def update_selected_query_nms(self, output, new_reference_points, topk_proposals, layer_id, class_unselected, spatial_shapes):
        """
                # ---------------- 多次使用nms，去除重合框  -----------------------
        """
        # if layer_id != self.num_layers - 1:    # 最后一层不进行nms
        th = 0.8
        iou_threshold = [0.8, 0.7, 0.6, 0.5, 0.8, 0.95]

        dn_number = new_reference_points.shape[0] - topk_proposals.shape[1]
        if dn_number > 0:
            dn_part_ref = new_reference_points[:dn_number, :, :]
            dn_output = output[:dn_number, :, :]
            dn_class = class_unselected[:dn_number, :, :]
            ref_ = new_reference_points[dn_number:, :, :]
            out_ = output[dn_number:, :, :]
            class_ = class_unselected[dn_number:, :, :]
            nms_s = self.nms_query_low_score(iou_threshold[layer_id], ref_.permute(1, 0, 2), class_.permute(1, 0, 2), spatial_shapes)
            # print("ori number:", ref_.shape[0],  "  remove query number:", ref_.shape[0] - nms_s.shape[1], "reserved number:", nms_s.shape[1])
            out_sd = torch.gather(out_, 0, nms_s.permute(1, 0).unsqueeze(-1).repeat(1, 1, 256))
            ref_sd = torch.gather(ref_, 0, nms_s.permute(1, 0).unsqueeze(-1).repeat(1, 1, 4))
            output = torch.cat([dn_output, out_sd], dim=0)
            new_reference_points = torch.cat([dn_part_ref, ref_sd])
        else:
            import time                
            t1 = time.time()
            nms_s = self.nms_query_low_score(iou_threshold[layer_id], new_reference_points.permute(1, 0, 2), class_unselected.permute(1, 0, 2), spatial_shapes)
            print("nms need time:", time.time() - t1)
            print("remove query number:", new_reference_points.shape[0] - nms_s.shape[1])
            output = torch.gather(output, 0, nms_s.permute(1, 0).unsqueeze(-1).repeat(1, 1, 256))
            new_reference_points = torch.gather(new_reference_points, 0, nms_s.permute(1, 0).unsqueeze(-1).repeat(1, 1, 4))

        return output, new_reference_points, nms_s, dn_number



    def binary_selected(self, output, layer_id, enc_outputs_class_unselected):
        binary = self.binary_embed[layer_id](self.norm(output)).sigmoid()
        print("dec binary max:", binary.max(), "  min:", binary.min())
        # class_pred = enc_outputs_class_unselected.max(-1)[0].sigmoid()
        # binary = class_pred.unsqueeze(-1) * binary    # 0.005

        B, N, C = binary.shape
        max_ = -1; number_list = []
        selected = []
        for i in range(B):
            tp = torch.where(binary[i, :, :].squeeze(-1) >= 0.03)
            # tp = torch.where(binary[i, :, :].squeeze(-1) >= 0.1)

            # tp = torch.where(binary[i, :, :].squeeze(-1) >= 0.001)

            # tp = torch.where(class_pred[i, :] >= 0.01)
            number_list.append(tp[0].shape[0])
            if max_ < tp[0].shape[0]:
                max_ = tp[0].shape[0]
            selected.append(tp[0])

        # if max_ < 100:    # 设置最小query
        #     max_ = 100

        final_max = 0
        if max_ == 0:
            final_max = 100
        else:
            final_max = max_
        print("query num 2:", max_)

        # if max_ == 15070:
        #     print()

        max_scores = enc_outputs_class_unselected.max(-1)[0].sigmoid()
        topk_proposals = None
        for i in range(B):
            supp = final_max - selected[i].shape[0]
            if supp > 0:
                if max_ == 0:
                    value_, topk_pro = torch.topk(max_scores[i:i+1, :], supp, dim=1) # bs, nq
                else:
                    value_, topk_pro = torch.topk(-max_scores[i:i+1, :], supp, dim=1) # bs, nq
            else:
                topk_pro = None
            if topk_proposals is None:
                if topk_pro is not None:
                    topk_proposals = torch.cat([selected[i].unsqueeze(0), topk_pro], dim=1)
                else:
                    topk_proposals = selected[i].unsqueeze(0)
            else:
                if topk_pro is not None:
                    cur_ = torch.cat([selected[i].unsqueeze(0), topk_pro], dim=1)
                    topk_proposals = torch.cat([topk_proposals, cur_], dim=0)
                else:
                    cur_ = selected[i].unsqueeze(0)
                    topk_proposals = torch.cat([topk_proposals, cur_], dim=0)

        return topk_proposals 


    def binary_process(self, output, layer_id, new_reference_points, topk_proposals, enc_outputs_class_unselected):
        """
                # ---------------- 分类预测，去除重合框  -----------------------
        """

        dn_number = new_reference_points.shape[0] - topk_proposals.shape[1]
        if dn_number > 0:
            dn_part_ref = new_reference_points[:dn_number, :, :]
            dn_output = output[:dn_number, :, :]
            ref_ = new_reference_points[dn_number:, :, :]
            out_ = output[dn_number:, :, :]
            enc_ = enc_outputs_class_unselected[dn_number:, :, :]
            binary_s = self.binary_selected(out_.permute(1, 0, 2), layer_id, enc_.permute(1, 0, 2))
            print("ori number:", ref_.shape[0],  "  remove query number:", ref_.shape[0] - binary_s.shape[1], "reserved number:", binary_s.shape[1])
            out_sd = torch.gather(out_, 0, binary_s.permute(1, 0).unsqueeze(-1).repeat(1, 1, 256))
            ref_sd = torch.gather(ref_, 0, binary_s.permute(1, 0).unsqueeze(-1).repeat(1, 1, 4))
            output = torch.cat([dn_output, out_sd], dim=0)
            new_reference_points = torch.cat([dn_part_ref, ref_sd])
        else:
            import time                
            t1 = time.time()
            binary_s = self.binary_selected(output.permute(1, 0, 2), layer_id, enc_outputs_class_unselected.permute(1, 0, 2))
            print("binary need time:", time.time() - t1)
            print("remove query number:", new_reference_points.shape[0] - binary_s.shape[1])
            # output = torch.gather(output, 0, binary_s.permute(1, 0).unsqueeze(-1).repeat(1, 1, 256))
            output = torch.gather(output, 0, binary_s.permute(1, 0).unsqueeze(-1).repeat(1, 1, 256))
            new_reference_points = torch.gather(new_reference_points, 0, binary_s.permute(1, 0).unsqueeze(-1).repeat(1, 1, 4))

        return output, new_reference_points, binary_s, dn_number

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None, # num_queries, bs, 2
                # for memory
                level_start_index: Optional[Tensor] = None, # num_levels
                spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                valid_ratios: Optional[Tensor] = None,
                topk_proposals: Optional[Tensor] = None,
                tgt_undetach_class: Optional[Tensor] = None,
                tgt_undetach_binary: Optional[Tensor] = None,
                ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        output = tgt

        intermediate = []
        nms_selected = []
        dn_numbers = []
        topk_ps = topk_proposals
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]  

        for layer_id, layer in enumerate(self.layers):
            # preprocess ref points
            if self.training and self.decoder_query_perturber is not None and layer_id != 0:
                reference_points = self.decoder_query_perturber(reference_points)

            if self.deformable_decoder:
                if reference_points.shape[-1] == 4:
                    reference_points_input = reference_points[:, :, None] \
                                            * torch.cat([valid_ratios, valid_ratios], -1)[None, :] # nq, bs, nlevel, 4
                else:
                    assert reference_points.shape[-1] == 2
                    reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]
                query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :]) # nq, bs, 256*2 
            else:
                query_sine_embed = gen_sineembed_for_position(reference_points) # nq, bs, 256*2
                reference_points_input = None

            # conditional query
            raw_query_pos = self.ref_point_head(query_sine_embed) # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos
            if not self.deformable_decoder:
                query_sine_embed = query_sine_embed[..., :self.d_model] * self.query_pos_sine_scale(output)

            # modulated HW attentions
            if not self.deformable_decoder and self.modulate_hw_attn:
                refHW_cond = self.ref_anchor_head(output).sigmoid() # nq, bs, 2
                query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / reference_points[..., 2]).unsqueeze(-1)
                query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / reference_points[..., 3]).unsqueeze(-1)

            # random drop some layers if needed
            dropflag = False
            if self.dec_layer_dropout_prob is not None:
                prob = random.random()
                if prob < self.dec_layer_dropout_prob[layer_id]:
                    dropflag = True
            if not dropflag:
                output = layer(
                    tgt = output,
                    tgt_query_pos = query_pos,
                    tgt_query_sine_embed = query_sine_embed,
                    tgt_key_padding_mask = tgt_key_padding_mask,
                    tgt_reference_points = reference_points_input,

                    memory = memory,
                    memory_key_padding_mask = memory_key_padding_mask,

                    memory_level_start_index = level_start_index,
                    memory_spatial_shapes = spatial_shapes,
                    memory_pos = pos,

                    self_attn_mask = tgt_mask,
                    cross_attn_mask = memory_mask,

                    mlp_layer=self.mlp,
                    class_embed= self.class_embed1,
                    binary_embed = self.binary_embed,
                    norm=self.norm,
                    layer_id=layer_id,
                    tgt_undetach_class=tgt_undetach_class,
                    tgt_undetach_binary=tgt_undetach_binary,
                    class_fea=self.conditional_binary,
                    topk_proposals=topk_proposals,
                )

            # iter update
            if self.bbox_embed1 is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)

                # if self.conditional_binary is not None:
                #     binary = self.binary_embed[layer_id](self.norm(output)).sigmoid()
                #     # binary_ = (binary* 100).int()
                #     # n, b, _ = binary_.shape
                #     # weight_ = self.conditional_binary(binary_.view(-1))
                #     # output = output + weight_.view(n,b,-1) * binary

                #     output = output * (1 + binary)

                delta_unsig = self.bbox_embed1[layer_id](self.norm(output))
                # delta_unsig = self.pred_in_4_layers(self.bbox_embed1, self.bbox_embed2, layer_id, output, level_start_index, topk_proposals)

                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                # 暂时使用nms这个变量，方法采用binary
                if self.nms_decoder and layer_id == 2:
                    class_unselected = self.class_embed1[layer_id](self.norm(output))
                    output, new_reference_points, binary_s, dn_number = self.binary_process(output, layer_id, new_reference_points, topk_proposals, class_unselected)
                    topk_ps = binary_s
                    nms_selected.append(binary_s)
                    dn_numbers.append(dn_number)
                else:
                    nms_selected.append(0)
                    dn_numbers.append(0)

                # if self.nms_decoder and layer_id < 0:
                #     class_unselected = self.class_embed1[layer_id](self.norm(output))
                #     output, new_reference_points, nms_s, dn_number = self.update_selected_query_nms(output, new_reference_points, topk_ps, layer_id, class_unselected, spatial_shapes)

                #     # pos_sft = self.merge_query(new_reference_points)
                #     # output = (pos_sft@output.permute(1,0,2)).permute(1, 0, 2) + output

                #     topk_ps = nms_s
                #     nms_selected.append(nms_s)
                #     dn_numbers.append(dn_number)
                # else:
                #     nms_selected.append(0)
                #     dn_numbers.append(0)



                # select # ref points
                if self.dec_layer_number is not None and layer_id != self.num_layers - 1:
                    nq_now = new_reference_points.shape[0]
                    select_number = self.dec_layer_number[layer_id + 1]
                    if nq_now != select_number:
                        class_unselected = self.class_embed[layer_id](output) # nq, bs, 91
                        topk_proposals = torch.topk(class_unselected.max(-1)[0], select_number, dim=0)[1] # new_nq, bs
                        new_reference_points = torch.gather(new_reference_points, 0, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)) # unsigmoid

                if self.rm_detach and 'dec' in self.rm_detach:
                    reference_points = new_reference_points
                else:
                    reference_points = new_reference_points.detach()
                if self.use_detached_boxes_dec_out:
                    ref_points.append(reference_points)
                else:
                    ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))
            if self.dec_layer_number is not None and layer_id != self.num_layers - 1:
                if nq_now != select_number:
                    output = torch.gather(output, 0, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model)) # unsigmoid

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
            [id_selected for id_selected in nms_selected],
            [i for i in dn_numbers],
        ]

    def merge_query(self, new_reference_points):
        # class_ = torch.argmax(class_unselected.sigmoid(), dim=2)
        # batch = class_.shape[1]
        # cls_matrix = None
        # for i in range(batch):
        #     b = class_[:, i:i+1] - class_[:, i:i+1].T
        #     if cls_matrix is None:
        #         cls_matrix = b.unsqueeze(0)
        #     else:
        #         cls_matrix = torch.cat([cls_matrix, b.unsqueeze(0)], 0)

        # cls_matrix[cls_matrix != 0] = 1
        # cls_matrix = 1 - cls_matrix

        batch = new_reference_points.shape[1]
        pos_matrix = None
        for i in range(batch):
            x = new_reference_points[:, i:i+1, 0] - new_reference_points[:, i:i+1, 0].T
            y = new_reference_points[:, i:i+1, 1] - new_reference_points[:, i:i+1, 1].T
            x = torch.sqrt(x*x + y*y)
            del y
            w = new_reference_points[:, i:i+1, 2] - new_reference_points[:, i:i+1, 2].T
            h = new_reference_points[:, i:i+1, 3] - new_reference_points[:, i:i+1, 3].T
            w = torch.abs(w) + torch.abs(h)
            del h
            if pos_matrix is None:
                pos_matrix = (x + w).unsqueeze(0)
            else:
                pos_matrix = torch.cat([pos_matrix, (x + w).unsqueeze(0)], 0)
            del x, w


        pos_sft = torch.softmax(pos_matrix, dim=-1)
        return pos_sft * pos_sft.shape[1]


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 add_channel_attention=False,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 ):
        super().__init__()
        # self attention
        if use_deformable_box_attn:
            self.self_attn = MSDeformableBoxAttention(d_model, n_levels, n_heads, n_boxes=n_points, used_func=box_attn_type)
        else:
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # channel attention
        self.add_channel_attention = add_channel_attention
        if add_channel_attention:
            self.activ_channel = _get_activation_fn('dyrelu', d_model=d_model)
            self.norm_channel = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        # channel attn
        if self.add_channel_attention:
            src = self.norm_channel(src + self.activ_channel(src))

        return src

class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 key_aware_type=None,
                 decoder_sa_type='ca',
                 module_seq=['sa', 'ca', 'ffn'],
                 ):
        super().__init__()
        self.module_seq = module_seq
        assert sorted(module_seq) == ['ca', 'ffn', 'sa']

        # cross attention
        if use_deformable_box_attn:
            self.cross_attn = MSDeformableBoxAttention(d_model, n_levels, n_heads, n_boxes=n_points, used_func=box_attn_type)
        else:
            self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = None

        # self.external_att = External_attention(256)
        self.external_att_mh = External_attention_mh(256)
        self.external_att = self.external_att_mh

        # self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm22 = nn.LayerNorm(d_model)
        self.linear_tgt = nn.Linear(d_model, d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_type = key_aware_type
        self.key_aware_proj = None
        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']

        if decoder_sa_type == 'ca_content':
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt




    def forward_sa(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None, # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None, # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None, # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None, # num_levels
                memory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None, # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None, # mask used for cross-attention
                mlp_layer: Optional[Tensor] = None, 
                class_embed: Optional[Tensor] = None,
                binary_embed: Optional[Tensor] = None,
                norm: Optional[Tensor] = None,
                layer_id: Optional[Tensor] = None,
                tgt_undetach_class: Optional[Tensor] = None,
                tgt_undetach_binary: Optional[Tensor] = None,
                class_fea: Optional[Tensor] = None,
                topk_proposals: Optional[Tensor] = None,
            ):
        # self attention
        # if self.self_attn is not None:
        if True:
            if self.decoder_sa_type == 'sa':
                # q = k = self.with_pos_embed(tgt, tgt_query_pos)
                # tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
                # tgt = tgt + self.dropout2(tgt2)
                # tgt = self.norm2(tgt)

                # query = self.with_pos_embed(tgt, tgt_query_pos)
                # # query = mlp_layer(query)
                # query_sum = torch.sum(query, dim=2)
                # ratio = tgt.shape[0] / 300.0
                # tgt2 = torch.softmax(query_sum, dim=0).unsqueeze(-1) * tgt * ratio

                # ratio = tgt.shape[0] / 300.0
                # tgt2 = torch.softmax(cls_, dim=0).unsqueeze(-1) * tgt.shape[2] * ratio
                # print(cls_.max(), "over 0.1:", cls_[cls_ > 0.1].shape[0], " ori:", cls_.shape[0])
                # query = self.with_pos_embed(tgt, tgt_query_pos)
                # query = mlp_layer(query)



                if layer_id == 0:
                    # print(tgt_undetach_binary.shape)
                    cls_ = tgt_undetach_binary.sigmoid().permute(1, 0, 2)
                    # cls2 = tgt_undetach_class.max(-1)[0].sigmoid().unsqueeze(-1).permute(1, 0, 2)
                else:
                    # cls2 = class_embed[layer_id - 1](norm(tgt)).max(-1)[0].sigmoid().unsqueeze(-1)
                    cls_ = binary_embed[layer_id - 1](norm(tgt)).sigmoid()
                # print("mean:", cls_.mean(dim=0))
                cls_ = cls_.detach() #* cls2.detach()
                # cls_ = torch.sqrt(cls_ * cls2)
                # tgt1 = self.linear_tgt(self.with_pos_embed(tgt, tgt_query_pos)) * (cls_)
                # tgt1 = self.linear_tgt(tgt) * (cls_)
                tgt1 = tgt * ((cls_ - cls_.min()) / (cls_.max() - cls_.min() + 0.000001))
                tgt1 = tgt + self.dropout2(tgt1)
                tgt1 = self.norm2(tgt1)

                b, n = topk_proposals.shape
                n_, b, c = tgt.shape
                num_dn = n_ - n

                self.external_att = self.external_att_mh
                if num_dn > 0:
                    tgt_dn = self.external_att(tgt1[:num_dn, :, :])
                    tgt_ = self.external_att(tgt1[num_dn:, :, :])
                    tgt1 = torch.cat([tgt_dn, tgt_], dim=0)
                else:
                    tgt1 = self.external_att(tgt1)

                tgt = tgt + self.dropout2(tgt1)
                tgt = self.norm22(tgt)

                pass
            elif self.decoder_sa_type == 'ca_label':
                bs = tgt.shape[1]
                k = v = self.label_embedding.weight[:, None, :].repeat(1, bs, 1)
                tgt2 = self.self_attn(tgt, k, v, attn_mask=self_attn_mask)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            elif self.decoder_sa_type == 'ca_content':
                tgt2 = self.self_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                            tgt_reference_points.transpose(0, 1).contiguous(),
                            memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index, memory_key_padding_mask).transpose(0, 1)
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            else:
                raise NotImplementedError("Unknown decoder_sa_type {}".format(self.decoder_sa_type))

        return tgt

    def forward_ca(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None, # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None, # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None, # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None, # num_levels
                memory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None, # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None, # mask used for cross-attention
            ):
        # cross attention
        if self.key_aware_type is not None:

            if self.key_aware_type == 'mean':
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == 'proj_mean':
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError("Unknown key_aware_type: {}".format(self.key_aware_type))
        # tgt2 = self.cross_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
        #                        tgt_reference_points.transpose(0, 1).contiguous(),
        #                        memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index, memory_key_padding_mask).transpose(0, 1)

        # tgt2 = self.cross_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
        #                        tgt_reference_points.transpose(0, 1).contiguous(),
        #                        self.with_pos_embed(memory, memory_pos).transpose(0, 1), memory_spatial_shapes, memory_level_start_index, memory_key_padding_mask).transpose(0, 1)


        tgt2 = self.cross_attn(tgt.transpose(0, 1),
                               tgt_reference_points.transpose(0, 1).contiguous(),
                               memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index, memory_key_padding_mask).transpose(0, 1)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        return tgt

    def forward(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None, # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None, # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None, # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None, # num_levels
                memory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None, # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None, # mask used for cross-attention
                mlp_layer: Optional[Tensor] = None,
                class_embed: Optional[Tensor] = None,
                binary_embed: Optional[Tensor] = None,
                norm: Optional[Tensor] = None,
                layer_id: Optional[Tensor] = None,
                tgt_undetach_class: Optional[Tensor] = None,
                tgt_undetach_binary: Optional[Tensor] = None,
                class_fea: Optional[Tensor] = None,
                topk_proposals:Optional[Tensor] = None,
            ):

        for funcname in self.module_seq:
            if funcname == 'ffn':
                tgt = self.forward_ffn(tgt)
            elif funcname == 'ca':
                tgt = self.forward_ca(tgt, tgt_query_pos, tgt_query_sine_embed, \
                    tgt_key_padding_mask, tgt_reference_points, \
                        memory, memory_key_padding_mask, memory_level_start_index, \
                            memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask)
            elif funcname == 'sa':
                tgt = self.forward_sa(tgt, tgt_query_pos, tgt_query_sine_embed, \
                    tgt_key_padding_mask, tgt_reference_points, \
                        memory, memory_key_padding_mask, memory_level_start_index, \
                            memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask, mlp_layer, class_embed, binary_embed, norm, layer_id, tgt_undetach_class, tgt_undetach_binary, class_fea, topk_proposals)
            else:
                raise ValueError('unknown funcname {}'.format(funcname))

        return tgt


def _get_clones(module, N, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_deformable_transformer(args):
    decoder_query_perturber = None
    if args.decoder_layer_noise:
        from .utils import RandomBoxPerturber
        decoder_query_perturber=RandomBoxPerturber(
                x_noise_scale=args.dln_xy_noise, y_noise_scale=args.dln_xy_noise, 
                w_noise_scale=args.dln_hw_noise, h_noise_scale=args.dln_hw_noise)

    use_detached_boxes_dec_out = False
    try:
        use_detached_boxes_dec_out = args.use_detached_boxes_dec_out
    except:
        use_detached_boxes_dec_out =False

    return DeformableTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_unicoder_layers=args.unic_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=args.query_dim,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
        modulate_hw_attn=True,

        deformable_encoder=True,
        deformable_decoder=True,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        use_deformable_box_attn=args.use_deformable_box_attn,
        box_attn_type=args.box_attn_type,

        learnable_tgt_init=True,
        decoder_query_perturber=decoder_query_perturber,

        add_channel_attention=args.add_channel_attention,
        add_pos_value=args.add_pos_value,
        random_refpoints_xy=args.random_refpoints_xy,

        # two stage
        two_stage_type=args.two_stage_type,  # ['no', 'standard', 'early']
        two_stage_pat_embed=args.two_stage_pat_embed,
        two_stage_add_query_num=args.two_stage_add_query_num,
        two_stage_learn_wh=args.two_stage_learn_wh,
        two_stage_keep_all_tokens=args.two_stage_keep_all_tokens,
        dec_layer_number=args.dec_layer_number,
        rm_self_attn_layers=None,
        key_aware_type=None,
        layer_share_type=None,

        rm_detach=None,
        decoder_sa_type=args.decoder_sa_type,
        module_seq=args.decoder_module_seq,

        embed_init_tgt=args.embed_init_tgt,
        use_detached_boxes_dec_out=use_detached_boxes_dec_out,

        # gqs gcs
        gcs_enc = args.gcs_enc,
        gcs_transformer_enc = args.gcs_transformer_enc,
        gqs_use = args.use_gqs_transformer,

        # cross_coding
        cross_coding = args.cross_coding,
        nms_free = args.nms_th_free,
        nms_decoder = args.nms_decoder,
        class_num = args.num_classes,
        binary = args.binary_loss,
    )


