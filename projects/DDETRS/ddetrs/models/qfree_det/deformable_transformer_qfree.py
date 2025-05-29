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

from .util import box_ops

from .util.misc import inverse_sigmoid
from .utils import gen_encoder_output_proposals, MLP,_get_activation_fn, gen_sineembed_for_position
from .ops.modules import MSDeformAttn

import torch.nn.functional as F
from .util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, compute_iou, box_iou


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
                one2many = 2,
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
        self.one2many = one2many

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
            decoder_layer_one2one = DeformableTransformerDecoderLayer_one2one(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points, use_deformable_box_attn=use_deformable_box_attn, box_attn_type=box_attn_type,
                                                          key_aware_type=key_aware_type,
                                                          decoder_sa_type=decoder_sa_type,
                                                          module_seq=module_seq)

            decoder_layer_12many = DeformableTransformerDecoderLayer_12many(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points, use_deformable_box_attn=use_deformable_box_attn, box_attn_type=box_attn_type,
                                                          key_aware_type=key_aware_type,
                                                          decoder_sa_type=decoder_sa_type,
                                                          module_seq=module_seq)

        else:
            raise NotImplementedError

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer_one2one, decoder_layer_12many, num_decoder_layers, decoder_norm,
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
                                        one2many=self.one2many,
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

        self.enc_out_class_embed1 = None
        self.enc_out_bbox_embed1 = None
        self.enc_out_class_embed2 = None
        self.enc_out_bbox_embed2 = None
        self.enc_binary_class = None
        self.conditional_binary_enc = None
        self.enc_binary_value_fc = None

        self.enc_exsa = None
       
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


    def select_topk(self, selected, binary_score, max_, numbers=2000):
        out = []
        batch = len(selected)
        for i in range(batch):
            data = selected[i]
            binary_ = binary_score[i]
            length_ = data.shape[0]
            if length_ > numbers:
                max_ = numbers
                binarys = torch.gather(binary_, 0, data.unsqueeze(-1)) 
                id_sort_value, sort_indice = torch.sort(binarys, dim=0, descending=True)   # score 排序
                ori_id = torch.gather(data, 0, sort_indice[:, 0])
                out.append(ori_id[:numbers])

            else:
                out.append(data)
        
        return out, max_


    def class_selected(self, enc_outputs_class_unselected):
        class_pred = enc_outputs_class_unselected.max(-1)[0].sigmoid()

        B, N = class_pred.shape
        max_ = -1; number_list = []
        selected = []
        for i in range(B):
            tp = torch.where(class_pred[i, :] >= 0.05)

            number_list.append(tp[0].shape[0])
            if max_ < tp[0].shape[0]:
                max_ = tp[0].shape[0]
            selected.append(tp[0])

        selected, max_final = self.select_topk(selected, class_pred.unsqueeze(-1), max_, numbers=900)

        final_max = 0
        if max_ <= 300:
            final_max = 300
        else:
            final_max = max_final

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



    def forward(self, srcs, masks, refpoint_embed, pos_embeds, tgt, attn_mask=None, dn_meta=None):
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
            output_memory, output_proposals = gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes, None)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            
            binary_fea = output_memory

            if self.enc_out_class_embed2 is None:
                enc_outputs_class_unselected = self.enc_out_class_embed1(output_memory)

            if self.enc_out_bbox_embed2 is None:
                enc_outputs_coord_unselected = self.enc_out_bbox_embed1(binary_fea) + output_proposals # (bs, \sum{hw}, 4) unsigmoid

            encoder_feas_box = None

            topk_proposals = self.class_selected(enc_outputs_class_unselected)

            # gather boxes
            refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)) # unsigmoid
            refpoint_embed_ = refpoint_embed_undetach.detach()
            init_box_proposal = torch.gather(output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).sigmoid() # sigmoid

            # gather tgt
            tgt_undetach_class = torch.gather(enc_outputs_class_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 1))

            src_flatten_return = None
            src_flatten_coord = None

            # gather tgt
            if self.embed_init_tgt:
                tgt_undetach = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))
                tgt_undetach_bias_4 = None
            else:
                tgt_undetach = torch.gather(binary_fea, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))
                tgt_undetach_bias_4 = None

            if self.embed_init_tgt:
                tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1) # nq, bs, d_model
                tgt_undetach_bias_4 = None
            else:
                tgt_ = tgt_undetach.detach()


            if refpoint_embed is not None:
                refpoint_embed=torch.cat([refpoint_embed,refpoint_embed_],dim=1)

                tgt_undetach_class = None
                tgt_undetach_binary = None

                tgt=torch.cat([tgt,tgt_],dim=1)
            else:
                tgt_undetach_class = None
                tgt_undetach_binary = None
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
                tgt_undetach_binary=tgt_undetach_binary,
                dn_meta=dn_meta,)
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

        gcs_transformer_out = None        
        gcs_back_out = None

        return hs, references, nms_selected, dn_numbers, hs_enc, gcs_transformer_out, gcs_back_out, gqs_enc, ref_gqs_enc, ref_enc, init_box_proposal, tgt_undetach_bias_4, src_flatten_return, src_flatten_coord, level_start_index, spatial_shapes, memory, topk_proposals, encoder_feas_box, mask_flatten, valid_ratios, src_flatten



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

    def __init__(self, decoder_one2one, decoder_12many, num_layers, norm=None, 
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
                    one2many=2,
                    ):
        super().__init__()
        self.one2many = one2many
        if num_layers > 0:
            self.layers_12many = _get_clones(decoder_12many, one2many, layer_share=dec_layer_share)
            self.layers_one2one = _get_clones(decoder_one2one, 6-one2many, layer_share=dec_layer_share)
            self.layers_12many.extend(self.layers_one2one)
            self.layers = self.layers_12many
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

        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None

        if rm_dec_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError

        self.bbox_embed1 = None
        self.class_embed1 = None
        self.bbox_embed2 = None
        self.class_embed2 = None
        self.binary = binary
        self.mlp = None  
        self.binary_embed = None
        self.conditional_binary = None
        self.dec_binary_value_fc = None

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
                dn_meta: Optional[Tensor] = None,
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
            else:
                reference_points_input = None

            # random drop some layers if needed
            dropflag = False
            if self.dec_layer_dropout_prob is not None:
                prob = random.random()
                if prob < self.dec_layer_dropout_prob[layer_id]:
                    dropflag = True
            if not dropflag:
                if layer_id == self.one2many:
                    output = output.detach()
                
                output = layer(
                    tgt = output,
                    tgt_query_pos = None,
                    tgt_query_sine_embed = None,
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
                    dn_meta=dn_meta,
                    ref_points=ref_points,
                )

 
            # iter update
            if self.bbox_embed2 is None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed1[layer_id](self.norm(output))
                
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()



            nms_selected.append(0)
            dn_numbers.append(0)

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


class DeformableTransformerDecoderLayer_12many(nn.Module):
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

                topk_proposals: Optional[Tensor] = None,
            ):
        # cross attention
        if self.key_aware_type is not None:

            if self.key_aware_type == 'mean':
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == 'proj_mean':
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError("Unknown key_aware_type: {}".format(self.key_aware_type))

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
                dn_meta: Optional[Tensor] = None,
                ref_points: Optional[Tensor] = None,
            ):

        self.module_seq = ['ca', 'ffn']  # only ca

        for funcname in self.module_seq:
            if funcname == 'ffn':
                tgt = self.forward_ffn(tgt)
            elif funcname == 'ca':
                tgt = self.forward_ca(tgt, tgt_query_pos, tgt_query_sine_embed, \
                    tgt_key_padding_mask, tgt_reference_points, \
                        memory, memory_key_padding_mask, memory_level_start_index, \
                            memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask, topk_proposals)
            else:
                raise ValueError('unknown funcname {}'.format(funcname))

        return tgt


class DeformableTransformerDecoderLayer_one2one(nn.Module):
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
        #self.module_seq = ['ca', 'sa', 'ffn']
        assert sorted(module_seq) == ['ca', 'ffn', 'sa']

        # final 2sa
        self.module_seq = ['ca', 'sa', 'ffn', 'sa3', 'ffn3']

        d_ffn = 2048

        # cross attention
        if "ca" in self.module_seq:
            if use_deformable_box_attn:
                self.cross_attn = MSDeformableBoxAttention(d_model, n_levels, n_heads, n_boxes=n_points, used_func=box_attn_type)
            else:
                self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
            self.dropout1 = nn.Dropout(dropout)
            self.norm1 = nn.LayerNorm(d_model)


        # ffn
        if "ffn" in self.module_seq:
            self.linear1 = nn.Linear(d_model, d_ffn)
            self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
            self.dropout3 = nn.Dropout(dropout)
            self.linear2 = nn.Linear(d_ffn, d_model)
            self.dropout4 = nn.Dropout(dropout)
            self.norm3 = nn.LayerNorm(d_model)

        # sa
        if "sa" in self.module_seq:
            self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.norm2 = nn.LayerNorm(d_model)


        # sa3
        if "sa3" in self.module_seq:
            self.self_attn3 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.norm5 = nn.LayerNorm(d_model)

        # ffn3
        if "ffn3" in self.module_seq:
            self.linear31 = nn.Linear(d_model, d_ffn)
            self.dropout31 = nn.Dropout(dropout)
            self.linear32 = nn.Linear(d_ffn, d_model)
            self.dropout32 = nn.Dropout(dropout)
            self.norm33 = nn.LayerNorm(d_model)


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


    def forward_ffn3(self, tgt):
        tgt2 = self.linear32(self.dropout31(self.activation(self.linear31(tgt))))

        tgt = tgt + self.dropout32(tgt2)
        tgt = self.norm33(tgt)
        return tgt

    def get_dynamic_attn_mask(self, dn_meta, query_num):
        dn_number = dn_meta['num_dn_group']
        pad_size = dn_meta['pad_size']
        single_pad = dn_meta['single_pad']

        tgt_size = query_num
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
        return attn_mask

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
                dn_meta: Optional[Tensor] = None,
                ref_points: Optional[Tensor] = None,
            ):
        # self attention
        # if self.self_attn is not None:
        if True:
            if self.decoder_sa_type == 'sa':
                if dn_meta is not None:
                    self_attn_mask = self.get_dynamic_attn_mask(dn_meta, tgt.shape[0])
                else:
                    self_attn_mask = None

                q = k = tgt
                tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)

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

        return tgt, tgt2

    def forward_sa3(self,
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
                dn_meta: Optional[Tensor] = None,
                ref_points: Optional[Tensor] = None,
            ):
        # self attention
        # if self.self_attn is not None:
        if True:
            if self.decoder_sa_type == 'sa':
                if dn_meta is not None:
                    self_attn_mask = self.get_dynamic_attn_mask(dn_meta, tgt.shape[0])
                else:
                    self_attn_mask = None

                q = k = tgt
                tgt2 = self.self_attn3(q, k, tgt, attn_mask=self_attn_mask)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm5(tgt)
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

                topk_proposals: Optional[Tensor] = None,
            ):
        # cross attention
        if self.key_aware_type is not None:

            if self.key_aware_type == 'mean':
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == 'proj_mean':
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError("Unknown key_aware_type: {}".format(self.key_aware_type))

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
                dn_meta: Optional[Tensor] = None,
                ref_points: Optional[Tensor] = None,
            ):

        for funcname in self.module_seq:
            if funcname == 'ffn':
                tgt = self.forward_ffn(tgt)
            elif funcname == 'ffn3':
                tgt = self.forward_ffn3(tgt)

            elif funcname == 'ca':
                tgt = self.forward_ca(tgt, tgt_query_pos, tgt_query_sine_embed, \
                    tgt_key_padding_mask, tgt_reference_points, \
                        memory, memory_key_padding_mask, memory_level_start_index, \
                            memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask, topk_proposals)

            elif funcname == 'sa':
                tgt, tgt2 = self.forward_sa(tgt, tgt_query_pos, tgt_query_sine_embed, \
                    tgt_key_padding_mask, tgt_reference_points, \
                        memory, memory_key_padding_mask, memory_level_start_index, \
                            memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask, mlp_layer, class_embed, binary_embed, norm, layer_id, tgt_undetach_class, tgt_undetach_binary, class_fea, topk_proposals, dn_meta, ref_points)


            elif funcname == 'sa3':
                tgt = self.forward_sa3(tgt, tgt_query_pos, tgt_query_sine_embed, \
                    tgt_key_padding_mask, tgt_reference_points, \
                        memory, memory_key_padding_mask, memory_level_start_index, \
                            memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask, mlp_layer, class_embed, binary_embed, norm, layer_id, tgt_undetach_class, tgt_undetach_binary, class_fea, topk_proposals, dn_meta, ref_points)
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

        one2many = args.ONE_TO_MANY_LAYER,
    )


