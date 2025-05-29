_base_ = ['coco_transformer.py']

num_classes=91
use_gqs_transformer = False
gcs_enc = False
gcs_transformer_enc = False

use_xywh_loss = True
reweight_small = False
loss_backbone = False
cross_coding = False
boost_loss = False
match = False

binary_loss = False
query_loss = False
unmatch_wh = False    
nms_th_free = 0.5    
nms_decoder = False    


unmatch_encoder_wh = False

two_stage_bbox_embed_share = False
two_stage_class_embed_share = False

dec_pred_bbox_embed_share = True 
dec_pred_class_embed_share = True 

ONE_TO_MANY_LAYER =  4    
TWO_LEVEL_PREDICT = False   
compile2 = False
compile_para = 'default'   
match_cost_opt = False    


num_queries = 300 

num_select = 300  
nms_iou_threshold = -1 
dn_number = 100    

set_cost_class = 2.0  
set_cost_bbox = 5.0 
set_cost_giou = 2.0   
set_cost_binary = 0.0  
bbox_loss_coef = 5.0   
giou_loss_coef = 2.0   

cls_loss_coef = 2.0     
interm_loss_coef = 1.0  

two_stage_keep_all_tokens = False  

lr = 0.0001
param_dict_type = 'default'
lr_backbone = 1e-05
lr_backbone_names = ['backbone.0']
lr_linear_proj_names = ['reference_points', 'sampling_offsets']
lr_linear_proj_mult = 0.1
ddetr_lr_param = False
batch_size = 2    #  2
weight_decay = 0.0001
epochs = 12
lr_drop = 11
save_checkpoint_interval = 1
clip_max_norm = 0.1
onecyclelr = False
multi_step_lr = False
lr_drop_list = [33, 45]


modelname = 'dino'
frozen_weights = None
backbone = 'swin_T_224_1k'  # 'swin_S_224_22k'  # swin_L_384_22k
use_checkpoint = False
backbone_dir = "./pretrained"

dilation = False
position_embedding = 'sine'
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None

enc_layers = 6
dec_layers = 6
only_decoder = False

unic_layers = 0
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8

query_dim = 4
num_patterns = 0
pdetr3_bbox_embed_diff_each_layer = False
pdetr3_refHW = -1
random_refpoints_xy = False
fix_refpoints_hw = -1
dabdetr_yolo_like_anchor_update = False
dabdetr_deformable_encoder = False
dabdetr_deformable_decoder = False
use_deformable_box_attn = False
box_attn_type = 'roi_align'
dec_layer_number = None
num_feature_levels = 4    # default 4
enc_n_points = 4
dec_n_points = 4
decoder_layer_noise = False
dln_xy_noise = 0.2
dln_hw_noise = 0.2
add_channel_attention = False
add_pos_value = False
two_stage_type = 'standard'
two_stage_pat_embed = 0
two_stage_add_query_num = 0

two_stage_learn_wh = False
two_stage_default_hw = 0.05


transformer_activation = 'relu'
batch_norm_type = 'FrozenBatchNorm2d'
masks = False
aux_loss = True



mask_loss_coef = 1.0
dice_loss_coef = 1.0


no_interm_box_loss = False
focal_alpha = 0.25

decoder_sa_type = 'sa' # ['sa', 'ca_label', 'ca_content']
matcher_type = 'HungarianMatcher' # or SimpleMinsumMatcher
decoder_module_seq = ['sa', 'ca', 'ffn']


# for dn
use_dn = True

dn_box_noise_scale = 0.4
dn_label_noise_ratio = 0.5
embed_init_tgt = True
dn_labelbook_size = 91

match_unstable_error = True

# for ema
use_ema = False
ema_decay = 0.9997
ema_epoch = 0

use_detached_boxes_dec_out = False

