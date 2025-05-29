

import torch
from torch import nn, Tensor
from .qfree_det.ops.modules import MSDeformAttn

class DecoderForGenerate(nn.Module):
    def __init__(self):
        super().__init__()

        self.cross_attn = MSDeformAttn(256, 4, 8, 16)
        self.norm_ca = nn.LayerNorm(256)
    
        d_model = 256
        d_ffn = 2048 # 768
        dropout = 0.1

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, query, out_qfree, memory, mask_flatten, level_start_index, spatial_shapes, valid_ratios, dn_meta):
        reference_points = out_qfree['pred_boxes'].permute(1, 0, 2)
        reference_points_input = reference_points[:, :, None] \
                                            * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
        
        # tgt2 = self.cross_attn(query.transpose(0, 1),
        #                        reference_points_input.transpose(0, 1).contiguous(),
        #                        memory.transpose(0, 1), spatial_shapes, level_start_index, mask_flatten).transpose(0, 1)
        if dn_meta is not None:
            pad_size = dn_meta["pad_size"]
            # for dn
            if "dn_meta" in out_qfree:
                ref_points = dn_meta["output_known_lbs_bboxes"]["pred_boxes"].permute(1, 0, 2)
            else:
                ref_points = dn_meta["output_known_lbs_bboxes"]["aux_outputs"][-1]["pred_boxes"].permute(1, 0, 2)
            ref_points_input_dn = ref_points[:, :, None] \
                                            * torch.cat([valid_ratios, valid_ratios], -1)[None, :]

            tgt_dn = self.cross_attn(query[:, :pad_size, :],
                               ref_points_input_dn.transpose(0, 1).contiguous(),
                               memory, spatial_shapes, level_start_index, mask_flatten).transpose(0, 1)
            tgt2 = self.cross_attn(query[:, pad_size:, :],
                               reference_points_input.transpose(0, 1).contiguous(),
                               memory, spatial_shapes, level_start_index, mask_flatten).transpose(0, 1)
            tgt2 = torch.cat([tgt_dn, tgt2], dim=0)
        else:
            tgt2 = self.cross_attn(query,
                               reference_points_input.transpose(0, 1).contiguous(),
                               memory, spatial_shapes, level_start_index, mask_flatten).transpose(0, 1)

        query = query.permute(1, 0, 2) + tgt2
        query = self.norm_ca(query)

        query = self.forward_ffn(query)

        return query.permute(1, 0, 2)


    def forward_both_dn(self, query, out_qfree, memory, mask_flatten, level_start_index, spatial_shapes, valid_ratios, dn_meta):
        reference_points = out_qfree['pred_boxes'].permute(1, 0, 2)
        reference_points_input = reference_points[:, :, None] \
                                            * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
        
        tgt2 = self.cross_attn(query,
                               reference_points_input.transpose(0, 1).contiguous(),
                               memory, spatial_shapes, level_start_index, mask_flatten).transpose(0, 1)

        query = query.permute(1, 0, 2) + tgt2
        query = self.norm_ca(query)

        query = self.forward_ffn(query)

        return query.permute(1, 0, 2)



    
    def forward_enc_back(self, query, out_qfree, memory, mask_flatten, level_start_index, spatial_shapes, valid_ratios, dn_meta, pred_box):
        pred_box = pred_box.unsqueeze(0)
        query = query.unsqueeze(0)
        reference_points = pred_box.permute(1, 0, 2)
        reference_points_input = reference_points[:, :, None] \
                                            * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
        
        tgt2 = self.cross_attn(query,
                               reference_points_input.transpose(0, 1).contiguous(),
                               memory, spatial_shapes, level_start_index, mask_flatten).transpose(0, 1)

        query = query.permute(1, 0, 2) + tgt2
        query = self.norm_ca(query)

        query = self.forward_ffn(query)

        return query.permute(1, 0, 2)[0]


    def forward_both_dn_enc_back(self, query, out_qfree, memory, mask_flatten, level_start_index, spatial_shapes, valid_ratios, dn_meta, pred_box):
        reference_points = pred_box.permute(1, 0, 2)
        reference_points_input = reference_points[:, :, None] \
                                            * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
        
        tgt2 = self.cross_attn(query,
                               reference_points_input.transpose(0, 1).contiguous(),
                               memory, spatial_shapes, level_start_index, mask_flatten).transpose(0, 1)

        query = query.permute(1, 0, 2) + tgt2
        query = self.norm_ca(query)

        query = self.forward_ffn(query)

        return query.permute(1, 0, 2)



    