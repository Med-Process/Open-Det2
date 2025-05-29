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

import cv2
from .util import box_ops
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .util.box_iou_my import bbox_iou_my

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss)

from .deformable_transformer_qfree import build_deformable_transformer

from .utils import sigmoid_focal_loss, MLP, MLP_BBox

from .registry import MODULE_BUILD_FUNCS
from .dn_components import prepare_for_cdn,dn_post_process, dn_post_process_append, dn_post_process_append_zero_pad

from .utils import sigmoid_focal_loss, MLP, token_sigmoid_binary_focal_loss, sigmoid_loss_align_iou_score

from .util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, compute_iou, box_iou


from .ops.modules import MSDeformAttn

from .util.slconfig import DictAction, SLConfig
from .t5 import Generate_with_T5
from ..decoder_generate import DecoderForGenerate

USE_CLS_CLS = False   


class MLP_Projection(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.drop = nn.Dropout(0.1)
        h = [hidden_dim] * (num_layers - 1)
        self.linear = nn.Linear(256, 768, bias=False)
        self.linear_gating = nn.Linear(256, 768, bias=False)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        identity = self.linear(x)
        identity = identity * torch.sigmoid(self.linear_gating(x))

        for i, layer in enumerate(self.layers):
            x = self.drop(F.relu6(layer(x))) if i < self.num_layers - 1 else layer(x)
        return self.drop(x) + identity



class Classification_Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(256, 8)
        self.norm = nn.LayerNorm(256)
        self.drop = nn.Dropout()

    def get_dynamic_attn_mask(self, dn_meta, query_num):
        dn_number = dn_meta['num_dn_group']
        pad_size = dn_meta['pad_size']
        single_pad = dn_meta['single_pad']

        # tgt_size = pad_size + query_num
        tgt_size = query_num
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True
        return attn_mask

    def forward(self, x, dn_meta, drop=0.1):
        if dn_meta is not None:
            self_attn_mask = self.get_dynamic_attn_mask(dn_meta, x.shape[1])

        q = k = x.permute(1, 0, 2)
        if dn_meta is not None:
            x1 = self.self_attn(q, k, x.permute(1, 0, 2), attn_mask=self_attn_mask)[0].permute(1, 0, 2)
        else:
            x1 = self.self_attn(q, k, x.permute(1, 0, 2), attn_mask=None)[0].permute(1, 0, 2)
        x = x + self.drop(x1)
        x = self.norm(x)
        return x


class VL_Align_Ours(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.DYHEAD.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        # dot product soft token head
        
        self.dot_product_projection_image = nn.Linear(cfg.MODEL.DDETRS.HIDDEN_DIM, cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM, bias=True) # nn.Identity()
        self.dot_product_projection_back = nn.Linear(cfg.MODEL.DDETRS.HIDDEN_DIM, cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM, bias=True) # nn.Identity()
        self.back_dec = nn.Linear(768*2, 768)

        self.text_embedding_low = nn.Linear(768, 256)
        self.img_256 = nn.Linear(256, 256)

        self.fusion = MLP_Projection(256, 2048, cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM, 3)
        self.fusion_enc = MLP_Projection(256, 2048, cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM, 3)
        self.fusion_back = MLP_Projection(256, 2048, cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM, 3)

        self.norm_fuse = nn.LayerNorm(cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM)

        self.dec_gen_align = DecoderForGenerate()

        self.linear_dec = nn.Linear(cfg.MODEL.DDETRS.HIDDEN_DIM, cfg.MODEL.DDETRS.HIDDEN_DIM)

        self.norm_dec = nn.LayerNorm(cfg.MODEL.DDETRS.HIDDEN_DIM)

        self.activation = nn.GELU()

        self.dot_product_projection_text = nn.Identity() #nn.Linear(cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM, cfg.MODEL.DDETRS.HIDDEN_DIM, bias=True) # 768 -> 256
        self.log_scale = nn.Parameter(torch.Tensor([cfg.MODEL.DYHEAD.LOG_SCALE]), requires_grad=True)
        self.bias_lang = nn.Parameter(torch.zeros(cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM), requires_grad=True) # (768，)
        self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True) # size (1,)
    
    def vl_down_mapping(self, img, text):
        new_img = self.img_256(img)
        new_text = self.text_embedding_low(text)

        return new_img, new_text

    def vl_mapping(self, src_flatten, hs_enc, img):
        """
        backbone fea; encoder fea; decoder query
        """
       
        img = self.fusion(img)

        return img

    def vl_mapping_back_enc(self, back_query, enc_query):
        back_mapping = self.fusion_back(back_query)
        enc_mapping = self.fusion_enc(enc_query)
        return back_mapping, enc_mapping


    def forward(self, src, enc, x, text_embedding, text_mask):
        """
        x: visual features (bs, num_query, 256)
        embedding: language features (bs, max_num_object, 768)
        """
     
        dot_product_proj_queries = self.dot_product_projection_image(x)

        low_text = self.text_embedding_low(text_embedding)

        dot_low = torch.matmul(self.img_256(x), low_text.transpose(-1, -2))

       
        
        img_embedding = dot_product_proj_queries / dot_product_proj_queries.norm(dim=-1, keepdim=True)
       
        dot_product_logit = torch.matmul(img_embedding, text_embedding.transpose(-1, -2))

        
        dot_product_logit = dot_product_logit + dot_low


        B, N, C = dot_product_logit.shape
        new_text_mask = text_mask.unsqueeze(1).repeat(1, N, 1)

        dot_product_logit = dot_product_logit * new_text_mask

        return dot_product_logit
    


    def forward_dec(self, src, enc, x, text_embedding, text_mask, boxes, memory, mask_flatten, level_start_index, spatial_shapes, valid_ratios, dn_meta):
        """
        x: visual features (bs, num_query, 256)
        embedding: language features (bs, max_num_object, 768)
        """
        
        x = self.dec_gen_align.forward_both_dn(x, boxes, memory, mask_flatten, level_start_index, spatial_shapes, valid_ratios, dn_meta)
        dot_product_proj_queries = self.dot_product_projection_image(x)

        
        low_text = self.text_embedding_low(text_embedding)

        dot_low = torch.matmul(self.img_256(x), low_text.transpose(-1, -2))
        
        img_embedding = dot_product_proj_queries / dot_product_proj_queries.norm(dim=-1, keepdim=True)
        
        dot_product_logit = torch.matmul(img_embedding, text_embedding.transpose(-1, -2)) + dot_low

        B, N, C = dot_product_logit.shape
        new_text_mask = text_mask.unsqueeze(1).repeat(1, N, 1)


        dot_product_logit = dot_product_logit * new_text_mask

        return dot_product_logit


class Still_Classifier_Ori(nn.Module):
    def __init__(self, hidden_dim, num_class=1):
        super().__init__()
        self.body_one2many = nn.Linear(hidden_dim, hidden_dim)

        self.body_one2one = nn.Linear(hidden_dim, hidden_dim)
    
        self.init_weight()
    
    def init_weight(self):
        nn.init.xavier_uniform_(self.body_one2many.weight, gain=1)
        nn.init.constant_(self.body_one2many.bias, 0)

        nn.init.xavier_uniform_(self.body_one2one.weight, gain=1)
        nn.init.constant_(self.body_one2one.bias, 0)

    def forward(self, x, level=0):
        if level < 4:
            cls = self.body_one2many(x)
            return cls.mean(dim=-1, keepdim=True)
        else:
            cls = self.body_one2one(x)
            return cls.mean(dim=-1, keepdim=True)


class QFreeDet(nn.Module):
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
                    one2many=2,
                    two_level=False,

                    cfg = None,
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
        self.one2many = one2many
        self.two_level = two_level

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

        _class_embed = VL_Align_Ours(cfg)
         
        _bbox_embed = MLP_BBox(hidden_dim, hidden_dim, 4, 3)

        if 'flan-t5' in cfg.MODEL.TEXT.TEXT_DECODER:
            self.class_generate = Generate_with_T5(cfg=cfg)

        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if cfg.MODEL.USE_IOU_BRANCH:
            iou_head = nn.Linear(hidden_dim, hidden_dim)
            iou_head.bias.data = torch.ones(hidden_dim) * bias_value
            iou_head = [iou_head for i in range(transformer.num_decoder_layers)]

            self.iou_head = nn.ModuleList(iou_head)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
            for i in range(self.one2many):    
                if i == 0:
                    box_embed_layerlist[0] = copy.deepcopy(_bbox_embed)
                else:
                    box_embed_layerlist[i] = box_embed_layerlist[0]
        else:
            box_embed_layerlist = [copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)]
        if dec_pred_class_embed_share:
            class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
            for i in range(self.one2many):
                if i == 0:
                    class_embed_layerlist[0] = copy.deepcopy(_class_embed)
                else:
                    class_embed_layerlist[i] = class_embed_layerlist[0]

        else:
            class_embed_layerlist = [copy.deepcopy(_class_embed) for i in range(transformer.num_decoder_layers)]
        
        if not self.two_level:
            self.bbox_embed1 = nn.ModuleList(box_embed_layerlist)
            self.class_embed1 = nn.ModuleList(class_embed_layerlist)
        
            self.transformer.decoder.bbox_embed1 = self.bbox_embed1
            self.transformer.decoder.class_embed1 = self.class_embed1
        else:
            self.bbox_embed1 = nn.ModuleList(box_embed_layerlist)
            self.class_embed1 = nn.ModuleList(class_embed_layerlist)
            self.bbox_embed2 = nn.ModuleList(copy.deepcopy(box_embed_layerlist))
            self.class_embed2 = nn.ModuleList(copy.deepcopy(class_embed_layerlist))

            self.transformer.decoder.bbox_embed1 = self.bbox_embed1
            self.transformer.decoder.class_embed1 = self.class_embed1
            self.transformer.decoder.bbox_embed2 = self.bbox_embed2
            self.transformer.decoder.class_embed2 = self.class_embed2


        self.conditional_binary_enc = None
        self.transformer.conditional_binary_enc = self.conditional_binary_enc

        # two stage
        self.two_stage_type = two_stage_type
        self.two_stage_add_query_num = two_stage_add_query_num
        assert two_stage_type in ['no', 'standard'], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type != 'no':
            if two_stage_bbox_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                if not self.two_level:
                    self.transformer.enc_out_bbox_embed1 = copy.deepcopy(_bbox_embed)
                    self.transformer.backbone_out_bbox_embed1 = copy.deepcopy(_bbox_embed)
                else:
                    self.transformer.enc_out_bbox_embed1 = copy.deepcopy(_bbox_embed)
                    self.transformer.backbone_out_bbox_embed1 = copy.deepcopy(_bbox_embed)
                    self.transformer.enc_out_bbox_embed2 = copy.deepcopy(_bbox_embed)
                    self.transformer.backbone_out_bbox_embed2 = copy.deepcopy(_bbox_embed)
    
            if two_stage_class_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share

                self.transformer.enc_out_class_embed = Still_Classifier_Ori(hidden_dim)
            else:
                if not self.two_level:
                    self.transformer.enc_out_class_embed1 = Still_Classifier_Ori(hidden_dim)    # 原方法
                    self.transformer.enc_vl_alignment = VL_Align_Ours(cfg)
                else:
                    self.transformer.enc_out_class_embed1 = Still_Classifier_Ori(hidden_dim)    # 原方法
                    self.transformer.enc_out_class_embed2 = Still_Classifier_Ori(hidden_dim)    # 原方法

    
            self.refpoint_embed = None
            if self.two_stage_add_query_num > 0:
                self.init_ref_points(two_stage_add_query_num)

        self.class_embed_for_test = Still_Classifier_Ori(hidden_dim)

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
        from .util.visualizer import COCOVisualizer
        
        vslzr = COCOVisualizer()
        vslzr.visualize(samples.tensors[0].cpu(), targets[0], savedir="./out_fig", dpi=100)
        print("save data!...")


    def forward(self, samples: NestedTensor, targets:List=None, clip_object_descriptions_features=None, use_iou_branch=False, text_mask=None):
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

        hs, reference, nms_selected, dn_numbers, hs_enc, gcs_transformer_out, gcs_back_out, gqs_enc, ref_gqs_enc, ref_enc, init_box_proposal, tgt_undetach_bias_4, src_flatten_return, src_flatten_coord, level_start_index, spatial_shapes, memory, topk_proposals, encoder_feas_box, mask_flatten, valid_ratios, src_flatten = self.transformer(srcs, masks, input_query_bbox, poss,input_query_label,attn_mask, dn_meta=dn_meta)

        # In case num object=0
        hs[0] += self.label_enc.weight[0,0]*0.0

        # deformable-detr-like anchor update
        outputs_coord_list = []
        output_wh_list = []
        bbox_fea_list = []    
        binary_outputs_class = []
        if use_iou_branch:
            outputs_ious = []

        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs, nms_sd, dns) in enumerate(zip(reference[:-1], self.bbox_embed1, hs, nms_selected, dn_numbers)):
            
            if USE_CLS_CLS:
                layer_delta_unsig, bbx_fea = layer_bbox_embed(layer_hs, fea_return=True)
                bbox_fea_list.append(bbx_fea)
            elif self.two_level:    
                layer_delta_unsig = self.pred_in_4_layers(self.bbox_embed1, self.bbox_embed2, dec_lid, layer_hs, level_start_index, topk_proposals)
            else:
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
    
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)

            if self.wh_part:
                output_wh_list.append(layer_outputs_unsig[:, :, 2:])

            if use_iou_branch:

                pred_iou = self.iou_head[dec_lid](hs[dec_lid]).mean(dim=-1, keepdim=True)
                outputs_ious.append(pred_iou)

            binary_class = self.class_embed_for_test(layer_hs, dec_lid)
            binary_outputs_class.append(binary_class)


        if self.binary:
            if not USE_CLS_CLS:
                outputs_binary = torch.stack([layer_cls_embed(layer_hs.detach()) for
                                     layer_cls_embed, layer_hs in zip(self.binary_embed, hs)])

            else:
                tp_class = torch.stack([layer_cls_embed(layer_hs) for
                                        layer_cls_embed, layer_hs in zip(self.class_embed, hs)])
                outputs_binary = torch.stack([layer_cls_embed(torch.cat([tp_class[i], bbox_feas], dim=2).detach()) for
                                     layer_cls_embed, bbox_feas, i in zip(self.binary_embed, bbox_fea_list, [0,1,2,3,4,5])])
            
        else:
            outputs_binary = None

        src_flatten_fea = torch.gather(src_flatten, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 256))

        if not self.nms_decoder:
            outputs_coord_list = torch.stack(outputs_coord_list)
            binary_outputs_class = torch.stack(binary_outputs_class)
            outputs_class = binary_outputs_class
            if self.two_level and self.training:
                outputs_class = torch.stack([self.pred_in_4_layers(self.class_embed1, self.class_embed2, i, hs[i], level_start_index, topk_proposals) for i in range(len(hs))])
            elif self.training:
                if dn_meta is not None:
              
                    outputs_class = torch.stack([layer_cls_embed(src_flatten_fea, hs_enc[0], layer_hs, clip_object_descriptions_features, text_mask) for
                                     layer_cls_embed, layer_hs in zip(self.class_embed1, hs)])


                    last_one = self.class_embed1[-1].forward_dec(src_flatten_fea, hs_enc[0], hs[-1], clip_object_descriptions_features, text_mask, {"pred_boxes":outputs_coord_list[-1]}, memory, mask_flatten, level_start_index, spatial_shapes, valid_ratios, dn_meta)
                    last_two = self.class_embed1[-2].forward_dec(src_flatten_fea, hs_enc[0], hs[-2], clip_object_descriptions_features, text_mask, {"pred_boxes":outputs_coord_list[-2]}, memory, mask_flatten, level_start_index, spatial_shapes, valid_ratios, dn_meta)

                    outputs_class[-1] = last_one
                    outputs_class[-2] = last_two


            if self.wh_part:
                output_wh_list = torch.stack(output_wh_list)        
        else:
            outputs_class = [layer_cls_embed(layer_hs) for layer_cls_embed, layer_hs in zip(self.class_embed, hs)]

        if use_iou_branch:
            outputs_iou = torch.stack(outputs_ious)
        else:
            outputs_iou = None


        if self.dn_number > 0 and dn_meta is not None:
            if not self.nms_decoder and not self.binary:
                if use_iou_branch:
                    outputs_class, outputs_coord_list, outputs_iou, binary_outputs_class = \
                        dn_post_process(outputs_class, outputs_coord_list, outputs_iou, binary_outputs_class, 
                                        dn_meta, self.aux_loss, self._set_aux_loss)
                    out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord_list[-1], \
                           'pred_boxious': outputs_iou[-1], 'binary_outputs_class': binary_outputs_class[-1]}
                else:
                    outputs_class, outputs_coord_list, binary_outputs_class = \
                        dn_post_process(outputs_class, outputs_coord_list, None, binary_outputs_class, 
                                        dn_meta, self.aux_loss, self._set_aux_loss_ori)
                    out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord_list[-1], \
                           'binary_outputs_class': binary_outputs_class[-1]}
            else:
                if dn_meta['pad_size'] > 0:
                    out_pred = dn_post_process_append(outputs_class, outputs_coord_list, outputs_binary, output_wh_list,
                                    dn_meta,self.aux_loss,self._set_aux_loss)

                    outputs_class = [outputs_class[i][:, dn_meta['pad_size']:, :] for i in range(len(outputs_class))]
                    outputs_coord_list = [outputs_coord_list[i][:, dn_meta['pad_size']:, :] for i in range(len(outputs_coord_list))]
                else:
                    out_pred = dn_post_process_append_zero_pad(outputs_class, outputs_coord_list, outputs_binary, output_wh_list,
                                    dn_meta,self.aux_loss,self._set_aux_loss)

                    outputs_class = [outputs_class[i][:, dn_meta['pad_size']:, :] for i in range(len(outputs_class))]
                    outputs_coord_list = [outputs_coord_list[i][:, dn_meta['pad_size']:, :] for i in range(len(outputs_coord_list))]
                out = out_pred[-1]
        else:
            out_pred = dn_post_process_append_zero_pad(outputs_class, outputs_coord_list, outputs_iou, outputs_binary, output_wh_list,
                            dn_meta,self.aux_loss,self._set_aux_loss)
            out = out_pred[-1]

        if self.binary:
            if self.dn_number > 0 and dn_meta is not None:

                out['pred_binary'] = outputs_binary[5][:, dn_meta['pad_size']:, :]
            else:

                out['pred_binary'] = outputs_binary[-1]

        if self.wh_part:
            if self.dn_number > 0 and dn_meta is not None:
                out['pred_wh'] = output_wh_list[:, :, dn_meta['pad_size']:, :][-1]
            else:
                out['pred_wh'] = output_wh_list[-1]



        if self.aux_loss:
            if not self.nms_decoder:
                if use_iou_branch:
                    out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list, outputs_iou, binary_outputs_class)
                else:
                    out['aux_outputs'] = self._set_aux_loss_ori(outputs_class, outputs_coord_list, binary_outputs_class)
            else:
                out['aux_outputs'] = out_pred[:-1]


            if self.binary:
                if self.dn_number > 0 and dn_meta is not None:
                    for i in range(len(out['aux_outputs'])):
                        out['aux_outputs'][i]['pred_binary'] = outputs_binary[i][:, dn_meta['pad_size']:, :]
                else:
                    for i in range(len(out['aux_outputs'])):
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
            if not self.two_level:
                interm_class = self.transformer.enc_out_class_embed1(hs_enc[-1])
                if targets is not None:
                    interm_vl = self.transformer.enc_vl_alignment(src_flatten_fea, hs_enc[-1], hs_enc[-1], clip_object_descriptions_features, text_mask)
            else:
                interm_class = self.pred_in_4_layers_enc(self.transformer.enc_out_class_embed1, \
                                                         self.transformer.enc_out_class_embed2, \
                                                         hs_enc[-1], level_start_index)

            if self.binary:
                if USE_CLS_CLS:
                    binary_interm = self.transformer.enc_binary_class(torch.cat([interm_class, encoder_feas_box], dim=2).detach())
                else:
                    binary_interm = self.transformer.enc_binary_class(hs_enc[-1].detach())
                out['interm_outputs']['pred_binary'] = binary_interm

            if targets is not None:
                out['interm_outputs'] = {'pred_logits': interm_vl, 'pred_boxes': interm_coord, "binary_outputs_class": interm_class}
            else:
                out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}

            if self.wh_part:
                out['interm_outputs']['pred_wh'] = interm_coord[:, :, 2:]

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

        out["clip_object_descriptions_features"] = clip_object_descriptions_features
        out["backbone_fea"] = src_flatten
        out["encoder_fea"] = hs_enc

        return out, hs, hs_enc, dn_meta, memory, mask_flatten, level_start_index, spatial_shapes, valid_ratios, src_flatten_fea, src_flatten



    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_iou, binary_outputs_class):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_boxious': c, 'binary_outputs_class': d}
                for a, b, c, d in zip(outputs_class[:-1], outputs_coord[:-1], outputs_iou[:-1], binary_outputs_class[:-1])]
    
    @torch.jit.unused
    def _set_aux_loss_ori(self, outputs_class, outputs_coord, binary_outputs_class):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'binary_outputs_class': d}
                for a, b, d in zip(outputs_class[:-1], outputs_coord[:-1], binary_outputs_class[:-1])]


    def _set_aux_loss_iou(self, outputs_iou):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_boxious': a}
                for a in zip(outputs_iou[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses, reweight, boost_loss, one2many, cost_opt):
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
        self.cost_opt = cost_opt
        if reweight:
            print(30*"===", "Using rewight for small objects!...")

        self.boost_loss = boost_loss
        self.one2many = one2many

        from .matcher import HungarianMatcher_One_to_Many
        self.matcher_one_to_many = HungarianMatcher_One_to_Many(cost_opt, 
            cost_class=0.2, cost_bbox=5.0, cost_giou=2.0, cost_binary=0.0,
            focal_alpha=2.0
        )


    def loss_labelsVL(self, outputs, targets, indices, num_boxes, clip_sim, **kwargs):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        num_classes = 1
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        ce_mask = torch.ones_like(src_logits, device=src_logits.device)

        target_classes_onehot = torch.zeros(src_logits.size(),
                                dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device) # (bs, num_query, C)
        # loop over batch size
        for batch_idx, (src_idxs, target_idxs) in enumerate(indices):
            # loop over objects in one image

            if target_idxs.max() + 1 > src_logits.shape[2]:
                target_idxs = target_idxs % (targets[batch_idx]["labels"].shape[0] / self.matcher_one_to_many.one_to_many)
                ce_mask[batch_idx, :, int((targets[batch_idx]["labels"].shape[0] / self.matcher_one_to_many.one_to_many)):] = 0
            else:
                ce_mask[batch_idx, :, targets[batch_idx]["labels"].shape[0]:] = 0


            target_classes_onehot[batch_idx, src_idxs, target_idxs.int()] = 1


        ce_mask = ce_mask.bool()

        target_classes_onehot = target_classes_onehot@clip_sim

        before = ce_mask.sum()
        batch = len(targets)
        for i in range(batch):
            pred_box = outputs['pred_boxes'][i]
            tar_box = targets[i]['boxes']

            N = ce_mask[i].shape[1]
            if N == tar_box.shape[0]:    # one-to-one
                iou, _ = self.compute_box_iou(box_ops.box_cxcywh_to_xyxy(pred_box),
                                        box_ops.box_cxcywh_to_xyxy(tar_box)) 
                ce_mask[i, :, :N][iou < 0.05] = False    


        loss_ce = token_sigmoid_binary_focal_loss(src_logits, target_classes_onehot, text_mask=ce_mask)
        

        losses = {'loss_ce': loss_ce}

        return losses


    def loss_labels(self, outputs, targets, indices, num_boxes, clip_sim, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs

        src_logits = outputs['binary_outputs_class']
        align_score = outputs['pred_logits']

        num_class = 1

        idx = self._get_src_permutation_idx(indices)    
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], num_class,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)


        target_classes_onehot = target_classes_onehot[:,:,:-1]

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        align_score_id = align_score[idx]
        cls_idx = torch.cat([cls for (src, cls) in indices])
        if cls_idx.max() + 1 > align_score.shape[2]:
            cls_idx = cls_idx % align_score.shape[2]
        align_score_id_select = torch.gather(align_score_id.cpu(), 1, cls_idx.unsqueeze(1).cpu()).squeeze(-1).cuda()
        loss_ce_align_class = sigmoid_loss_align_iou_score(src_logits, target_classes_onehot, num_boxes, target_boxes, src_boxes, target_classes_o, align_score_id_select, alpha=self.focal_alpha, gamma=2, idx=idx)
        loss_ce = loss_ce + loss_ce_align_class

        losses = {'loss_bce': loss_ce}

        return losses


    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, clip_sim):
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

    def compute_box_iou(self, inputs, targets):
        """Compute pairwise iou between inputs, targets
        Both have the shape of [N, 4] and xyxy format
        """
        area1 = box_ops.box_area(inputs)
        area2 = box_ops.box_area(targets)

        lt = torch.max(inputs[:, None, :2], targets[:, :2])  # [N,M,2]
        rb = torch.min(inputs[:, None, 2:], targets[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)        # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter
        iou_metric = inter / union
        iou_diag = torch.diag(iou_metric)
        return iou_metric, iou_diag

    def loss_boxes(self, outputs, targets, indices, num_boxes, clip_sim):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}

        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))


        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
        with torch.no_grad():
            losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
            losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        # box iou
        if 'pred_boxious' in outputs:
            with torch.no_grad():
                _, ious = self.compute_box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes),
                                        box_ops.box_cxcywh_to_xyxy(target_boxes))                    
            tgt_iou_scores = ious
            src_iou_scores = outputs['pred_boxious'] # [B, N, 1]
            src_iou_scores = src_iou_scores[idx]
            src_iou_scores = src_iou_scores.flatten(0)
            tgt_iou_scores = tgt_iou_scores.flatten(0)
            loss_boxiou = F.binary_cross_entropy_with_logits(src_iou_scores, tgt_iou_scores, reduction='mean')
        if 'pred_boxious' in outputs:
            losses['loss_boxiou'] = loss_boxiou
        else:
            losses['loss_boxiou'] = loss_bbox.new_zeros([1], dtype=torch.float32)[0]

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

    def get_loss(self, loss, outputs, targets, indices, num_boxes, clip_sim, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'labelsVL': self.loss_labelsVL,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, clip_sim, **kwargs)

    def forward(self, outputs, targets, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        indices_out = {}
        clip_text_features = outputs["clip_object_descriptions_features"]
        clip_sim = torch.matmul(clip_text_features, clip_text_features.transpose(-1, -2))
        clip_sim[clip_sim < 0.999] = 0.0
        clip_sim[clip_sim > 0.1] = 1.0

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device=next(iter(outputs.values())).device
        indices = self.matcher(outputs_without_aux, targets)
        indices_out["indices_pred"] = indices

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_ = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes_], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
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

            indices_out["indices_pred_dn"] = dn_pos_idx

            output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']

            l_dict = {}
            for loss in self.losses:
                if loss == "labelsVL" :  # labelsVL--    dn part不需要该计算
                    continue

                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}

                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar, clip_sim, **kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)

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

            regular_dn = None

            losses.update(l_dict)

        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, clip_sim))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                if idx < self.one2many:
                    indices_12many, new_targets, one_to_many = self.matcher_one_to_many(aux_outputs, targets)
                    indices_out["indices_aux_pred_" + str(idx)] = indices_12many
                else:
                    indices = self.matcher(aux_outputs, targets)
                    indices_out["indices_aux_pred_" + str(idx)] = indices
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
                    if idx < self.one2many:
                        l_dict = self.get_loss(loss, aux_outputs, new_targets, indices_12many, num_boxes*one_to_many, clip_sim, **kwargs)
                    else:
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, clip_sim, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                # for dn part
                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    indices_out["indices_aux_pred_dn_" + str(idx)] = dn_pos_idx
                    l_dict={}
                    for loss in self.losses:
                        if loss == "labelsVL":    # labelsVL--
                            continue

                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar, clip_sim,
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
            interm_outputs = outputs['interm_outputs']
            
            indices_interm = self.matcher(interm_outputs, targets)
            indices_out["indices_interm_pred"] = indices_interm

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
                l_dict = self.get_loss(loss, interm_outputs, targets, indices_interm, num_boxes, clip_sim, **kwargs)                
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
                l_dict = self.get_loss(loss, interm_outputs, targets, indices_interm, num_boxes, clip_sim, **kwargs)
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
                l_dict = self.get_loss(loss, backbone_outputs, targets, indices_interm, num_boxes, clip_sim, **kwargs)
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
                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, clip_sim, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses, indices_out


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


# @MODULE_BUILD_FUNCS.registe_with_name(module_name='qfree-det')
def _build_qfreedet(args, cfg):
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

    model = QFreeDet(
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
        one2many = args.ONE_TO_MANY_LAYER,
        two_level = args.TWO_LEVEL_PREDICT,
        cfg = cfg
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)

    # prepare weight dict
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef, "loss_bbox_12many": args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict['loss_giou_12many'] = args.giou_loss_coef
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
            'loss_bbox_12many': 1.0 if not no_interm_box_loss else 0.0,
            'loss_giou_12many': 1.0 if not no_interm_box_loss else 0.0,
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


    losses = ['labelsVL', 'labels', 'boxes']

    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses,
                             reweight=args.reweight_small,
                             boost_loss=args.boost_loss,
                             one2many=args.ONE_TO_MANY_LAYER,
                             cost_opt=args.match_cost_opt)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(num_select=args.num_select, nms_iou_threshold=args.nms_iou_threshold)}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors


class DictToAttr:
    def __init__(self, data):
        self.__dict__.update(data)

def build_qfree_det(config_file, cfg):
    cfg_file = SLConfig.fromfile(config_file)
    cfg_dict = cfg_file._cfg_dict.to_dict()
    cfg_dict["device"] = "cuda"
    cfg_dict["save_log"] = True
    cfg_dict["find_unused_params"] = True
    cfg_dict["dn_scalar"] = 100
    cfg_dict["embed_init_tgt"] = False
    cfg_dict["dn_label_coef"] = 1.0
    cfg_dict["dn_bbox_coef"] = 1.0
    cfg_dict["use_ema"] = False
    cfg_dict["dn_box_noise_scale"] = 1.0

    cfg_att = DictToAttr(cfg_dict)
    

    qfree_model, criterion, postprocessors = _build_qfreedet(cfg_att, cfg)
    print("QFree model has been build sucessed!...")

    return qfree_model, criterion, postprocessors