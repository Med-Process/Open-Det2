import torch
import torch.nn.functional as F
from torch import nn

from transformers import BertTokenizer, AutoModelForCausalLM, T5TokenizerFast
from ..text.modeling_t5 import T5Config, T5ForConditionalGeneration


from peft import LoraConfig, get_peft_model, TaskType


class Generate_with_T5(nn.Module):
    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
        cfg=None,
    ):
        super().__init__()

        t5_model = cfg.MODEL.TEXT.TEXT_DECODER
        generate_loss_weight = cfg.MODEL.TEXT.GENERATE_LOSS_WEIGHT
        use_focal_loss = cfg.MODEL.TEXT.USE_FOCAL_LOSS
        self.use_all_negative = cfg.MODEL.TEXT.USE_ALL_NEGATIVE

        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config
        )
        # if cfg.MODEL.TEXT.FIX_TEXT_DECODER:
        #     for name, param in self.t5_model.named_parameters():
        #         param.requires_grad = False

        self.droup_out = nn.Dropout(0.1)


        self.max_txt_len = max_txt_len

        # freeze
        if cfg.STAGE == 1:
            for p in self.t5_model.parameters():
                p.requires_grad_(False)
        else:
            for p in self.t5_model.parameters():
                p.requires_grad_(True)

        # if cfg.MODEL.TEXT.FIX_TEXT_DECODER:
        #     print("The T5 model is fixed!...")
        #     for name, param in self.t5_model.named_parameters():
        #         print(name)
        #         if name == "share.weight" or name == "lm_head.weight" or name == "decoder_embedding.weight":
        #             param.requires_grad = False

        self.t5_proj = nn.Linear(
            256, self.t5_model.config.hidden_size
        )
        self.generate_loss_weight = generate_loss_weight
        self.use_focal_loss=use_focal_loss

    def forward(self, object_features, object_descriptions, object_features_att_mask):
        if object_features.shape[2] == 256:
            inputs_t5 = self.t5_proj(object_features)
        else:
            inputs_t5 = object_features
        
        inputs_t5 = inputs_t5 / (inputs_t5.norm(dim=-1, keepdim=True) + 1e-6)

        inputs_t5 = self.droup_out(inputs_t5)

        atts_t5 = object_features_att_mask 
        
        output_tokens = self.t5_tokenizer(
            object_descriptions,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(object_features.device)

        encoder_atts = atts_t5 

        targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )
        inputs_embeds = inputs_t5 
        loss = {}
        outputs = self.t5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            decoder_attention_mask=output_tokens.attention_mask,
            return_dict=True,
            labels=targets,
            use_focal_loss=self.use_focal_loss,
        )
        t5_loss = {'t5_loss':outputs.loss * self.generate_loss_weight}
        loss.update(t5_loss)

        return loss

    @torch.no_grad()
    def text_decoder(
        self,
        text_decoder_inputs,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        object_features = text_decoder_inputs['object_features']

        if object_features.shape[2] == 256:
            inputs_t5 = self.t5_proj(object_features)
        else:
            inputs_t5 = object_features

        atts_t5 = text_decoder_inputs['atts_t5']


        encoder_atts = atts_t5 
        inputs_embeds = inputs_t5 
        
        outputs = self.t5_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_length,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_beams, #num_captions,
            output_scores=True,
            return_dict_in_generate=True,
        )
        output_text = self.t5_tokenizer.batch_decode(
             outputs.sequences, skip_special_tokens=True
        )
        if num_beams>1:
            output_sequences_scores = outputs.sequences_scores.sigmoid()
        else:
            scores = torch.stack(list(outputs.scores),dim=0) #[30, 900, 32128]
            log_probs = F.log_softmax(scores, dim=-1)
            top_logprobs, predicted_classes = log_probs.topk(1)
            top_logprobs = top_logprobs.transpose(1,0)
            indexes = outputs.sequences > 0 
            sum_top_logprobs = []
            for top_logprob, index in zip(top_logprobs, indexes):
                sum_top_logprobs.append(torch.sum(top_logprob[index[1:]], dim=0))
            output_sequences_scores = torch.tensor(sum_top_logprobs).to(log_probs.device) #[900]

        output_dict = {
                    'pred_object_descriptions': output_text,
                    'logprobs': output_sequences_scores,
                }

        return output_dict
