import os
import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from transformers.models.t5.modeling_t5 import T5LayerNorm
from src.modules.transforms.mel_spectrogram import NormalizeDB
from src.modules.common import ResidualVectorQuantize

def repeat_feat(x, repeat_num, target_len, dim=-1):
    assert dim == -1
    x = torch.repeat_interleave(x, repeat_num, dim=dim)
    cur_len = x.size(dim)
    if cur_len < target_len:
        x_pad = torch.stack([x[:, :, -1]] * (target_len - cur_len), dim=dim)
        x = torch.cat([x, x_pad], dim=dim)
    x = x[:, :, :target_len]
    return x

def check_nan(x, n):
    if torch.isnan(x).any():
        print('{} NAN'.format(n))
        return True
    if torch.isinf(x).any():
        print('{} inf'.format(n))
        return True
    return False


def clip_b16(hidden_states):
    # clamp inf values to enable fp16 training
    if hidden_states.dtype == torch.float16:
        clamp_value = torch.where(
            torch.isinf(hidden_states).any(),
            torch.finfo(hidden_states.dtype).max - 1000,
            torch.finfo(hidden_states.dtype).max,
        )
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
    return hidden_states


class Mel2resVQ2MelSEANet_hubert_f0_spk(nn.Module):
    def __init__(
        self, 
        n_mels=80,
        d_model=512,
        num_token=500,
        semantic_dim=512,
        timbre_dim=192,
        num_spk=5994,
        codebook_size=[1024, 1024, 500],  # residual, prosody, semantic
        codebook_num=[1, 1, 1],  
        codebook_dim=[256, 256, 512],    
        codebook_ratio = [25, 12.5, 25, 0.5],  # residual, prosody, semantic, timbre
        encoder=None,
        encoder1=None,
        encoder2=None,
        encoder3=None,
        decoder=None,
        decoder1=None,
        decoder2=None,
        decoder3=None,
        decoder_g=None,
        decoder_stream1=None,
        decoder_stream2=None,
        prosody_cls_type='conv2',
        vq_type='vq',
        pre_norm=False,
        codebook_norm=False,
        codebook_distance='l2',
        causal=False,
        detach_stream=[False, False],  # detach output from timbre, prosody
        **unused):
        super(Mel2resVQ2MelSEANet_hubert_f0_spk, self).__init__()
        self.causal = causal
        self.encoder = encoder
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        if encoder3 is None:
            self.encoder3 = encoder2
        else:
            self.encoder3 = encoder3        
        
        self.decoder = decoder
        self.decoder1 = decoder1 if decoder1 is not None else nn.Identity()
        self.decoder2 = decoder2
        self.decoder3 = decoder3
        self.decoder_g = decoder_g
        
        self.decoder_stream1 = decoder_stream1
        self.decoder_stream2 = decoder_stream2

        if pre_norm:
            self.pre_norm = T5LayerNorm(semantic_dim)
        else:
            self.pre_norm = nn.Identity()

        self.codebook_ratio = codebook_ratio
        self.d_model = d_model
        self.num_token = num_token
        self.semantic_dim = semantic_dim
        self.timbre_dim = timbre_dim
        self.detach_spk_emb = detach_stream[0]
        self.num_spk = num_spk
        self.n_mels = n_mels

        # VQ codebook
        self.layers = nn.ModuleList()
        for i in range(len(codebook_num) - 1):
            self.layers.append(ResidualVectorQuantize(codebook_num[i], input_dim=d_model, dim=codebook_dim[i], 
                                                        num_tokens=codebook_size[i], codebook_norm=codebook_norm,
                                                        distance=codebook_distance, vq_type=vq_type))
        sample_rate = 16000
        self.semantic_freq = codebook_ratio[2]
        self.prosody_freq = codebook_ratio[1]
        self.res_freq = codebook_ratio[0]
        self.timbre_freq = codebook_ratio[-1]
        self.segment_len = int(sample_rate / self.timbre_freq) 
        print("Semantic freq: {}, prosody freq: {}, res freq: {}".format(codebook_ratio[2], codebook_ratio[1], codebook_ratio[0]))
        print("Timbre freq: {}".format(self.timbre_freq))
        print('segment_len for spk emb: {}'.format(self.segment_len))

        # generate semantic token
        assert semantic_dim == codebook_dim[-1]
        assert num_token == codebook_size[-1]
        self.semantic_emb = nn.Embedding(num_token, semantic_dim)

        # prosody classifier and loss
        self.mel_norm = NormalizeDB(NormalizeDB(min_level_db=-100, max_abs_value=1.0, symmetric=True, normalization=True))
        self.pitch_pool = nn.AvgPool1d(2, 2, padding=0, ceil_mode=True)  # data input freq: 25 to 12.5Hz
        if prosody_cls_type == 'conv2':
            self.pitch_classifier = nn.Sequential(nn.Conv1d(d_model, d_model // 2, 3, 1, bias=True, padding=1),
                                                  nn.ELU(),
                                                  nn.Conv1d(d_model // 2, 1, 1, 1, bias=True))
            self.energy_classifier = nn.Sequential(nn.Conv1d(d_model, d_model // 2, 3, 1, bias=True, padding=1),
                                                   nn.ELU(),
                                                   nn.Conv1d(d_model // 2, 1, 1, 1, bias=True))
        elif prosody_cls_type == 'conv':
            self.pitch_classifier = nn.Conv1d(d_model, 1, 3, 1, bias=True, padding=1)
            self.energy_classifier = nn.Conv1d(d_model, 1, 3, 1, bias=True, padding=1)
        else:
            self.pitch_classifier = nn.Conv1d(d_model, 1, 1, 1, bias=True, padding=0)
            self.energy_classifier = nn.Conv1d(d_model, 1, 1, 1, bias=True, padding=0)

        self.emb_semantic_decoder = nn.Sequential(nn.Conv1d(self.timbre_dim + self.semantic_dim, self.d_model * 2, kernel_size=1, bias=True),
                                                   nn.ELU(), 
                                                   nn.Conv1d(self.d_model * 2, self.d_model, kernel_size=1, bias=True))
        

    def crop_mel(self, mel_recon, target_len):
        rec_len = mel_recon.size(-1)
        if target_len < rec_len:
            crop_l = (rec_len - target_len) // 2
            mel_recon = mel_recon[:, :, crop_l: crop_l + target_len]
        return mel_recon
    """
    该函数实现了从梅尔频谱图（mel）、语义令牌（semantic_token）、基频（f0）和说话人嵌入（spk_emb）生成重构的梅尔频谱图的功能。
    输入参数包括：
    - mel: 梅尔频谱图，形状为 [B, 1, T]
    - semantic_token: 语义令牌，用于提取语义特征
    - f0: 基频信息，可选参数
    - spk_emb: 说话人嵌入，可选参数
    
    函数首先提取语义特征，并将其与说话人嵌入进行拼接。然后，通过编码器和解码器对梅尔频谱图进行重构，并计算重构误差、基频预测误差和能量预测误差。
    最后，函数返回重构的梅尔频谱图以及各项损失值。
    """

    @torch.no_grad()
    def inference(self, mel, semantic_token, spk_emb=None, rec_mel=True, **unused):
        mel_ori_size = mel.size(-1)
        pad_size = 0
        
        spk_emb_norm = torch.norm(spk_emb, p=2, dim=-1, keepdim=True)
        spk_emb = spk_emb / (spk_emb_norm + 1e-6)
        
        semantic_feat = self.semantic_emb(semantic_token.detach())
        semantic_feat = self.pre_norm(semantic_feat)
        semantic_feat = semantic_feat.transpose(1, 2)
        # print(semantic_feat.size())
        acoustic_feat = mel
        # print(acoustic_feat.size())
        feat_layer1 = self.encoder(acoustic_feat)
        
        if self.causal:
            causal_mask = torch.tril(torch.ones([1, feat_layer1.size(-1), feat_layer1.size(-1)], device=feat_layer1.device))
        else:
            causal_mask = None
            
        feat_layer1 = self.encoder1(feat_layer1.transpose(1, 2), attention_mask=causal_mask)[0].transpose(1, 2)
        feat_layer1 = clip_b16(feat_layer1)

        if semantic_feat.size(-1) < feat_layer1.size(-1):  # 25Hz
            semantic_feat = torch.cat([semantic_feat] + [semantic_feat[:, :, -2:-1]] * int(feat_layer1.size(-1) - semantic_feat.size(-1)), dim=-1)

        feat_layer2 = self.encoder2(feat_layer1)
        # print(feat_layer1.size(), feat_layer2.size())

        timbre_global_feat = spk_emb.transpose(1, 2).detach()  # [B, C, T]
        if timbre_global_feat.size(-1) == 1:  # 25Hz
            expand_num = semantic_feat.size(-1)
        else:
            expand_num = int(self.semantic_freq / self.timbre_freq)
        timbre_global_feat = repeat_feat(timbre_global_feat, expand_num, semantic_feat.size(-1))

        timbre_time_feat_rec = torch.cat([semantic_feat, timbre_global_feat], dim=1) # [B, C, T]
        timbre_time_feat_rec = self.emb_semantic_decoder(timbre_time_feat_rec)
        timbre_time_feat_rec = self.decoder_g(timbre_time_feat_rec.transpose(1, 2), attention_mask=causal_mask)[0].transpose(1, 2)

        target_len = feat_layer1.size(-1)
        rec_len = timbre_time_feat_rec.size(-1)
        if target_len < rec_len:
            crop_l = (rec_len - target_len) // 2
            timbre_time_feat_rec = timbre_time_feat_rec[:, :, crop_l: crop_l + target_len]

        timbre_time_feat_rec_down = self.encoder3(timbre_time_feat_rec)  # 25->12.5hz
        feat_layer2_res = feat_layer2 - timbre_time_feat_rec_down.detach()
        timbre_time_feat_q, emo_vq_ind, quant_loss2 = self.layers[1](feat_layer2_res.transpose(2, 1))
        timbre_time_feat_q = timbre_time_feat_q.transpose(2, 1)

        timbre_time_feat_q = timbre_time_feat_q + timbre_time_feat_rec_down
        timbre_acous_rec = self.decoder(timbre_time_feat_q)
        # print(timbre_acous_rec.size())
        # print(timbre_acous_rec.size())
        target_len = feat_layer1.size(-1)
        rec_len = timbre_acous_rec.size(-1)
        if target_len < rec_len:
            crop_l = (rec_len - target_len) // 2
            timbre_acous_rec = timbre_acous_rec[:, :, crop_l: crop_l + target_len]
        elif target_len > rec_len:
            pad_len = target_len - rec_len
            timbre_acous_rec = torch.cat([timbre_acous_rec, timbre_acous_rec[:, :, -pad_len:]], dim=-1)

        timbre_acous_rec = self.decoder1(timbre_acous_rec)
        feat_layer1_res = feat_layer1 - timbre_acous_rec.detach()
        acoustic_feat_q, res_vq_ind, quant_loss1 = self.layers[0](feat_layer1_res.transpose(2, 1))  # quantize acoustic_res

        if rec_mel:
            acoustic_feat_q = acoustic_feat_q.transpose(2, 1)
            acoustic_feat_q = acoustic_feat_q + timbre_acous_rec

            acoustic_feat_q = self.decoder2(acoustic_feat_q.transpose(1, 2), attention_mask=causal_mask)[0].transpose(1, 2)  # 25->25Hz
            mel_recon = self.decoder3(acoustic_feat_q) # 25->100Hz

            target_len = mel.size(-1)
            rec_len = mel_recon.size(-1)
            if target_len < rec_len:
                crop_l = (rec_len - target_len) // 2
                mel_recon = mel_recon[:, :, crop_l: crop_l + target_len]
                # mel_recon_ori = mel_recon_ori[:, :, crop_l: crop_l + target_len]

            if pad_size > 0:
                mel_recon = mel_recon[:, :, :mel_ori_size]
                # mel_recon_ori = mel_recon_ori[:, :, :mel_ori_size]
        else:
            mel_recon = None
            # mel_recon_ori = None

        quan_loss = quant_loss1 + quant_loss2

        emo_vq_ind = emo_vq_ind[0]  # only one codebook
        res_vq_ind = res_vq_ind[0]  # only one codebook

        
        return mel_recon, quan_loss, res_vq_ind, emo_vq_ind #, mel_recon_ori


    @torch.no_grad()
    def generate(self, semantic_vq_ind, res_vq_ind, emo_vq_ind, spk_emb, **unused):
        # semantic: predicted 25Hz
        semantic_feat = self.semantic_emb(semantic_vq_ind.detach())
        semantic_feat = self.pre_norm(semantic_feat)
        semantic_feat = semantic_feat.transpose(1, 2)

        spk_emb_norm = torch.norm(spk_emb, p=2, dim=-1, keepdim=True)
        spk_emb = spk_emb / (spk_emb_norm + 1e-6)
        # timbre_global_feat = self.prompt_prenet(spk_emb)  # [B, T, C]
        # timbre_time_feat_rec, _ = self.emb_semantic_decoder(semantic_feat.transpose(1, 2), None, timbre_global_feat, None)
        # timbre_time_feat_rec = timbre_time_feat_rec.transpose(1, 2)
        
        if self.causal:
            causal_mask = torch.tril(torch.ones([1, semantic_vq_ind.size(-1), semantic_vq_ind.size(-1)], device=semantic_vq_ind.device))
        else:
            causal_mask = None

        timbre_global_feat = spk_emb.transpose(1, 2).detach()  # [B, C, T]
        if timbre_global_feat.size(-1) == 1:  # 25Hz
            expand_num = semantic_feat.size(-1)
        else:
            expand_num = int(self.semantic_freq / self.timbre_freq)
        timbre_global_feat = repeat_feat(timbre_global_feat, expand_num, semantic_feat.size(-1))

        timbre_time_feat_rec = torch.cat([semantic_feat, timbre_global_feat], dim=1) # [B, C, T]
        timbre_time_feat_rec = self.emb_semantic_decoder(timbre_time_feat_rec)
        seman_timbre_feat_rec = self.decoder_g(timbre_time_feat_rec.transpose(1, 2), attention_mask=causal_mask)[0].transpose(1, 2)
        seman_timbre_feat_rec_down = self.encoder3(seman_timbre_feat_rec)  # 25->12.5hz
        # feat_layer2_res = feat_layer2 - timbre_time_feat_rec.detach()
        # timbre_time_feat_q, emo_vq_ind, quant_loss2 = self.layers[1](feat_layer2_res.transpose(2, 1))
        timbre_time_feat_q = self.layers[1].get_embedding_from_code(emo_vq_ind.unsqueeze(0))
        timbre_time_feat_q = timbre_time_feat_q.transpose(2, 1)

        # reconstruct acoustic_feat from timbre_time 
        out_len = min(timbre_time_feat_q.size(-1), seman_timbre_feat_rec_down.size(-1))
        timbre_time_feat_q = timbre_time_feat_q[:, :, :out_len] + seman_timbre_feat_rec_down[:, :, :out_len]
        timbre_acous_rec = self.decoder(timbre_time_feat_q)  # 12.5Hz->25Hz
        out_len = min(timbre_acous_rec.size(-1), seman_timbre_feat_rec.size(-1))
        timbre_acous_rec = timbre_acous_rec[:, :, :out_len] 
        timbre_acous_rec = self.decoder1(timbre_acous_rec)
        acoustic_feat_q = self.layers[0].get_embedding_from_code(res_vq_ind.unsqueeze(0))
        acoustic_feat_q = acoustic_feat_q.transpose(2, 1)
        out_len = min(acoustic_feat_q.size(-1), timbre_acous_rec.size(-1))
        acoustic_feat_q = acoustic_feat_q[:, :, :out_len] + timbre_acous_rec[:, :, :out_len]

        acoustic_feat_q = self.decoder2(acoustic_feat_q.transpose(1, 2), attention_mask=causal_mask)[0].transpose(1, 2)  # 25->25Hz
        mel_recon = self.decoder3(acoustic_feat_q) # 25->100Hz
        
        return mel_recon

