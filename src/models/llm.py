from typing import Dict, Optional, Union, List
import math
from einops import rearrange
import torch
import torch.nn.functional as F
from torch import nn
from src.utils import ras_sampling

IGNORE_ID = -100


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        text_encoder_input_size: int,
        llm_input_size: int,
        llm_output_size: int,
        text_token_size: int,
        speech_token_size: int,
        text_encoder: torch.nn.Module,
        llm: torch.nn.Module,
        post_decoder: torch.nn.Module,
        spk_embed_dim: int = 192,
        spk_embed_dim_norm: bool = True,
        codebook_size=[32, 64, 512],  # residual, prosody, semantic codebook size
        codebook_dim=[64, 64, 512],   # residual, prosody, semantic codebook dim
        codebook_time_scale=[25, 12.5, 25, 0.5],  # Hz  residual,prosofy,semantic,timbre
        timbre_scale_bias = [10, 0.2],
        **unused
    ):
        super().__init__()
        self.llm_input_size = llm_input_size
            
        print('Norm spk emb: {}'.format(spk_embed_dim_norm))

        assert speech_token_size == codebook_size[-1]

        self.speech_token_size = speech_token_size
        # 1. build text token inputs related modules
        self.text_embedding = torch.nn.Embedding(text_token_size, text_encoder_input_size)
        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear(self.text_encoder.output_size(), llm_input_size)
        self.timbre_scale_bias = timbre_scale_bias
        self.post_decoder = post_decoder

        self.spk_embed_dim_norm = spk_embed_dim_norm
        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.stop_token = speech_token_size

        self.timbre_id_embedding = torch.nn.Embedding(1, spk_embed_dim)
        
        self.codebook_time_scale = codebook_time_scale
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.timbre_up_inter_ratio = int(self.codebook_time_scale[1] / self.codebook_time_scale[-1])

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size + 1, codebook_dim[-1])
        #  residual & prosody embedding
        self.residual_embedding = torch.nn.Embedding(codebook_size[0], codebook_dim[0])
        self.prosody_embedding = torch.nn.Embedding(codebook_size[1], codebook_dim[1])

        self.embed_affine_layer1 = torch.nn.Linear(codebook_dim[0] * 2 + codebook_dim[1] + codebook_dim[2] * 2, llm_input_size)
        self.embed_affine_layer2 = torch.nn.Linear(spk_embed_dim * 2, llm_input_size)

        decoder_out_dim = llm_output_size #codebook_dim[0] * 2 + codebook_dim[1] + codebook_dim[2] * 2
        self.llm_decoder = nn.Linear(llm_output_size + codebook_dim[-1] * 2, llm_output_size)

        self.llm_decoder_sem_in = nn.Linear(llm_output_size, llm_output_size * 2)
        self.llm_decoder_sem = nn.Linear(llm_output_size, speech_token_size + 1)
        
        self.llm_decoder_prosody_in = nn.Linear(codebook_dim[1], llm_output_size)
        self.llm_decoder_prosody = nn.Linear(llm_output_size, codebook_size[1])

        self.llm_decoder_res_in = nn.Linear(codebook_dim[0], llm_output_size)
        self.llm_decoder_res = nn.Linear(llm_output_size, codebook_size[0])
        

        self.llm_decoder_timbre = nn.Sequential(nn.Linear(decoder_out_dim, decoder_out_dim),
                                                nn.SiLU(),
                                                nn.Linear(decoder_out_dim, spk_embed_dim))

    def encode(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ):

        encoder_out, encoder_mask = self.text_encoder(
            text, text_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1
        )
        # print(2, encoder_out_lens)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens


    def generate_llm_input_inference(self, semantic_token, residual_token, prosody_token):
        # device = semantic_token.device
        # timbre_len = timbre_seq.size(-1)
        semantic_token_len = semantic_token.size(1)
        # prosody_token_len = prosody_token.size(1)
        input_prosody, input_semantic, input_residual = prosody_token, semantic_token, residual_token
        batch_size = 1

        if semantic_token_len % 2 == 1:
            input_semantic = semantic_token[:, :-1]
            input_residual = residual_token[:, :input_semantic.size(1)]

        semantic_token = self.speech_embedding(input_semantic.int())
        residual_token = self.residual_embedding(input_residual.int())
        prosody_token = self.prosody_embedding(input_prosody.int())

        semantic_token = rearrange(semantic_token, 'b (t n) c -> b t (n c)', n=2)
        residual_token = rearrange(residual_token, 'b (t n) c -> b t (n c)', n=2)

        if semantic_token.size(1) > prosody_token.size(1):
            pad_len = semantic_token.size(1) - prosody_token.size(1)
            prosody_token = torch.cat([prosody_token] + [prosody_token[:, -1, :].unsqueeze(1)] * int(pad_len), dim=1)
        elif semantic_token.size(1) < prosody_token.size(1):
            prosody_token = prosody_token[:, :semantic_token.size(1), :]

        # print(3, semantic_token.size(), residual_token.size(), prosody_token.size())
        speech_token = torch.cat([semantic_token, prosody_token, residual_token], dim=-1)
        speech_token = self.embed_affine_layer1(speech_token)
        llm_input = speech_token

        return llm_input

    def generate_llm_output(self, llm_output_seq, text_len, speech_token_len):
        # speech_token_len: prosody token len + timbre_len
        bz = llm_output_seq.size(0)

        out_speech, out_timbre = [], []
        for i in range(bz):
            # out_p_len_ = 0
            feat = llm_output_seq[i, text_len[i] + 1:text_len[i] + 1 + speech_token_len[i]] 
            out_speech.append(feat)

            # timbre_feat_ = torch.stack(timbre_feat_, dim=0)  # [T, C]
            # out_timbre.append(self.llm_decoder_timbre(timbre_feat_))

        out_speech = torch.cat(out_speech, dim=0).unsqueeze(0)    # [1, T, C]

        return out_speech #out_prosody, out_semantic, out_residual


    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            top_p: float = 0.8,
            top_k: int = 25,
            ignore_eos: bool = True,
    ):
        assert weighted_scores.dim() == 1
        if ignore_eos:
            weighted_scores = weighted_scores[:-1]
        top_ids = ras_sampling(weighted_scores, decoded_tokens, sampling, top_p, top_k)
        return top_ids

    @staticmethod
    def _repetition_penalty(logp, out_tokens, repetition_penalty, buffer_length):
        buffer_ids = torch.tensor(out_tokens[-buffer_length:], device=logp.device, dtype=torch.long).unsqueeze(0)
        rep_logp = torch.gather(logp, -1, buffer_ids)
        rep_logp = torch.where(rep_logp < 0, rep_logp * repetition_penalty, rep_logp)
        logp = logp.scatter(-1, buffer_ids, rep_logp)
        return logp

    @torch.inference_mode()
    def decoder_inference(self, lm_input, 
                          out_tokens_semantic, 
                          out_tokens_prosody,
                          out_tokens_residual,
                          *args,
                          **unused):
        att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=lm_input.device), torch.zeros(
            (0, 0, 0, 0), device=lm_input.device
        )
        sem1_id = out_tokens_semantic[-2]
        sem1_emb = self.speech_embedding.weight[sem1_id].reshape(1, -1)  # [1, C]
        sem2_id = out_tokens_semantic[-1]
        sem2_emb = self.speech_embedding.weight[sem2_id].reshape(1, -1)  # [1, C]
        lm_input = torch.cat([sem1_emb, sem2_emb, lm_input], dim=-1)  # [T, C + c]
        lm_input = self.llm_decoder(lm_input).unsqueeze(1)  # [T, 1, C]
        offset = 0
        for i in range(3):
            y_pred, att_cache, cnn_cache = self.post_decoder.forward_chunk(
                lm_input,
                offset=0,
                required_cache_size=-1,
                att_cache=att_cache,
                cnn_cache=cnn_cache,
                att_mask=torch.tril(
                    torch.ones(
                        (1, lm_input.shape[1], lm_input.shape[1]),
                        device=lm_input.device,
                    )
                ).to(torch.bool),
            )
            offset += 1
            # print(y_pred.size())
            y_pred = y_pred[:, -1]
            if i == 0:
                logits_prosody = self.llm_decoder_prosody(y_pred)
                last_prosody_token = logits_prosody.argmax(dim=-1)
                out_tokens_prosody.append(last_prosody_token.item())
                prosody_speech_feat = self.prosody_embedding.weight[last_prosody_token].reshape(1, 1, -1)
                lm_input = self.llm_decoder_prosody_in(prosody_speech_feat)
            else:
                logits_res = self.llm_decoder_res(y_pred)
                last_res_token = logits_res.argmax(dim=-1)
                out_tokens_residual.append(last_res_token.item())
                if i == 1:
                    res_speech_feat = self.residual_embedding.weight[last_res_token].reshape(1, 1, -1)
                    lm_input = self.llm_decoder_res_in(res_speech_feat)
                
        return out_tokens_prosody, out_tokens_residual
                

    @torch.inference_mode()
    def inference(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        prompt_semantic_token: torch.Tensor,
        prompt_residual_token: torch.Tensor,
        prompt_prosody_token: torch.Tensor,
        prompt_text: torch.Tensor = None,
        prompt_text_len: torch.Tensor = 0,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
        repetition_penalty: float = 1.0,
        buffer_length: int = 50,
        temperature: float = 1.0,
        top_p=0.8,
        top_k=25,
        **unused
    ) -> torch.Tensor:
        device = text.device
        if prompt_text is not None:
            text = torch.concat([prompt_text, text], dim=1)
            text_len += prompt_text_len

        assert text.size(0) == 1
        text = self.text_embedding(text)

        # 1. encode text
        text, text_len = self.encode(text, text_len)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        # if prompt_speech_token_len != 0:
        
        prompt_speech_token_emb = self.generate_llm_input_inference(prompt_semantic_token, prompt_residual_token, prompt_prosody_token)

        
        current_token_len = prompt_speech_token_emb.size(1)
        lm_input = torch.concat([sos_eos_emb, text, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. cal min/max_length
        min_len = math.ceil((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = math.ceil((text_len - prompt_text_len) * max_token_text_ratio)
        # print(min_len, max_len)
        # 5. step by step decode
        full_semantic_token_list = prompt_semantic_token.squeeze(0).tolist()
        out_tokens_semantic = []
        out_tokens_residual = []
        out_tokens_prosody = [] #[last_prosody_token]
        # out_timbre_seq = [] #[last_timbre_feat]
        offset = 0
        att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=lm_input.device), torch.zeros(
            (0, 0, 0, 0), device=lm_input.device
        )
        # print(min_len, max_len)
        while True:
            y_pred, att_cache, cnn_cache = self.llm.forward_chunk(
                lm_input,
                offset=0,
                required_cache_size=-1,
                att_cache=att_cache,
                cnn_cache=cnn_cache,
                att_mask=torch.tril(
                    torch.ones(
                        (1, lm_input.shape[1], lm_input.shape[1]),
                        device=lm_input.device,
                    )
                ).to(torch.bool),
            )
            logits_output = y_pred[:, -1]  # [1, C]
            current_token_len += 1

            # semantic
            out_semantic = self.llm_decoder_sem_in(logits_output)
            out_semantic = rearrange(out_semantic, 't (n c) -> t n c', n=2)
            
            logits_semantic = self.llm_decoder_sem(out_semantic)
            if temperature != 1:
                logits_semantic = logits_semantic / (temperature + 1e-5)
            logits_semantic = logits_semantic.log_softmax(dim=-1)
            logp = logits_semantic[:, 0]
            if repetition_penalty > 1.0 and len(out_tokens_semantic) > 0:
                logp = self._repetition_penalty(logp, full_semantic_token_list, repetition_penalty, buffer_length)
            top_ids = self.sampling_ids(logp.squeeze(dim=0),
                                        out_tokens_semantic,
                                        sampling,
                                        top_p, top_k,
                                        ignore_eos=True if len(out_tokens_semantic) < min_len else False).item()
            
            if top_ids == self.stop_token:
                break
            out_tokens_semantic.append(top_ids)
            full_semantic_token_list.append(top_ids)
            
            logp = logits_semantic[:, 1]
            if repetition_penalty > 1.0 and len(out_tokens_semantic) > 0:
                logp = self._repetition_penalty(logp, full_semantic_token_list, repetition_penalty, buffer_length)
            top_ids = self.sampling_ids(logp.squeeze(dim=0),
                                        out_tokens_semantic,
                                        sampling,
                                        top_p, top_k,
                                        ignore_eos=True if len(out_tokens_semantic) < min_len else False).item()
            
            if top_ids == self.stop_token:
                break
            out_tokens_semantic.append(top_ids)
            full_semantic_token_list.append(top_ids)
            
            out_tokens_prosody, \
            out_tokens_residual = self.decoder_inference(logits_output, 
                                                            out_tokens_semantic, 
                                                            out_tokens_prosody,
                                                            out_tokens_residual)
            if len(out_tokens_semantic) >= max_len:
                out_tokens_semantic = out_tokens_semantic[:max_len]
                out_tokens_residual = out_tokens_residual[:max_len]
                out_tokens_prosody = out_tokens_prosody[:math.ceil(max_len / 2)]
                break

            semantic_speech_feat = [self.speech_embedding.weight[out_tokens_semantic[-2]],
                                    self.speech_embedding.weight[out_tokens_semantic[-1]]]
            # print(semantic_speech_feat[0].size())
            semantic_speech_feat = torch.stack(semantic_speech_feat, 0).unsqueeze(0)
            # print(semantic_speech_feat.size())
            semantic_speech_feat = rearrange(semantic_speech_feat, 'b (t n) c -> b t (n c)', n=2)

            # prosody
            prosody_speech_feat = self.prosody_embedding.weight[out_tokens_prosody[-1]].reshape(1, 1, -1)
            
            # residual
            residual_speech_feat = [self.residual_embedding.weight[out_tokens_residual[-2]],
                                    self.residual_embedding.weight[out_tokens_residual[-1]]]
            # print(residual_speech_feat[0].size())
            residual_speech_feat = torch.stack(residual_speech_feat, 0).unsqueeze(0)
            # print(residual_speech_feat.size())
            residual_speech_feat = rearrange(residual_speech_feat, 'b (t n) c -> b t (n c)', n=2)

            speech_token = torch.cat([semantic_speech_feat, prosody_speech_feat, residual_speech_feat], dim=-1)
            speech_token = self.embed_affine_layer1(speech_token)

            offset += lm_input.size(1)
            lm_input = speech_token.reshape(1, 1, -1)
                

        semantic_token_output = torch.tensor([out_tokens_semantic], dtype=torch.int64, device=device)
        residual_token_output = torch.tensor([out_tokens_residual], dtype=torch.int64, device=device)
        prosody_token_output = torch.tensor([out_tokens_prosody], dtype=torch.int64, device=device)
        
        return semantic_token_output, residual_token_output, prosody_token_output


    @torch.inference_mode()
    def inference_condition(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        prompt_semantic_token: torch.Tensor,
        prompt_residual_token: torch.Tensor,
        prompt_prosody_token: torch.Tensor,
        prompt_text: torch.Tensor = None,
        prompt_text_len: torch.Tensor = 0,
        cond_semantic_token=None,
        **unused
    ) -> torch.Tensor:
        device = text.device
        if prompt_text is not None:
            text = torch.concat([prompt_text, text], dim=1)
            text_len += prompt_text_len

        assert text.size(0) == 1
        text = self.text_embedding(text)

        # 1. encode text
        text, text_len = self.encode(text, text_len)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        # if prompt_speech_token_len != 0:
        
        prompt_speech_token_emb = self.generate_llm_input_inference(prompt_semantic_token, prompt_residual_token, prompt_prosody_token)
        
        current_token_len = prompt_speech_token_emb.size(1)
        lm_input = torch.concat([sos_eos_emb, text, task_id_emb, prompt_speech_token_emb], dim=1)

        # 4. step by step decode
        out_tokens_semantic = []
        out_tokens_residual = []
        out_tokens_prosody = [] #[last_prosody_token]
        # out_timbre_seq = [] #[last_timbre_feat]
        offset = 0
        att_cache, cnn_cache = torch.zeros((0, 0, 0, 0), device=lm_input.device), torch.zeros(
            (0, 0, 0, 0), device=lm_input.device
        )
        # print(min_len, max_len)
        cond_i = 0
        cond_semantic_token = cond_semantic_token.squeeze()
        while True:
            y_pred, att_cache, cnn_cache = self.llm.forward_chunk(
                lm_input,
                offset=0,
                required_cache_size=-1,
                att_cache=att_cache,
                cnn_cache=cnn_cache,
                att_mask=torch.tril(
                    torch.ones(
                        (1, lm_input.shape[1], lm_input.shape[1]),
                        device=lm_input.device,
                    )
                ).to(torch.bool),
            )
            logits_output = y_pred[:, -1]  # [1, C]
            current_token_len += 1

            # semantic
            top_ids = cond_semantic_token[cond_i]
            out_tokens_semantic.append(top_ids)
            cond_i += 1
            if cond_i >= cond_semantic_token.size(0):
                break

            top_ids = cond_semantic_token[cond_i]
            out_tokens_semantic.append(top_ids)
            cond_i += 1
            
            out_tokens_prosody, \
            out_tokens_residual = self.decoder_inference(logits_output, 
                                                            out_tokens_semantic, 
                                                            out_tokens_prosody,
                                                            out_tokens_residual)
            if cond_i >= cond_semantic_token.size(0):
                out_tokens_semantic = out_tokens_semantic[:cond_semantic_token.size(0)]
                out_tokens_residual = out_tokens_residual[:cond_semantic_token.size(0)]
                out_tokens_prosody = out_tokens_prosody[:math.ceil(cond_semantic_token.size(0) / 2)]
                break

            semantic_speech_feat = [self.speech_embedding.weight[out_tokens_semantic[-2]],
                                    self.speech_embedding.weight[out_tokens_semantic[-1]]]
            # print(semantic_speech_feat[0].size())
            semantic_speech_feat = torch.stack(semantic_speech_feat, 0).unsqueeze(0)
            # print(semantic_speech_feat.size())
            semantic_speech_feat = rearrange(semantic_speech_feat, 'b (t n) c -> b t (n c)', n=2)

            # prosody
            prosody_speech_feat = self.prosody_embedding.weight[out_tokens_prosody[-1]].reshape(1, 1, -1)
            
            # residual
            residual_speech_feat = [self.residual_embedding.weight[out_tokens_residual[-2]],
                                    self.residual_embedding.weight[out_tokens_residual[-1]]]
            # print(residual_speech_feat[0].size())
            residual_speech_feat = torch.stack(residual_speech_feat, 0).unsqueeze(0)
            # print(residual_speech_feat.size())
            residual_speech_feat = rearrange(residual_speech_feat, 'b (t n) c -> b t (n c)', n=2)

            speech_token = torch.cat([semantic_speech_feat, prosody_speech_feat, residual_speech_feat], dim=-1)
            speech_token = self.embed_affine_layer1(speech_token)

            offset += lm_input.size(1)
            lm_input = speech_token.reshape(1, 1, -1)
            

        semantic_token_output = torch.tensor([out_tokens_semantic], dtype=torch.int64, device=device)
        residual_token_output = torch.tensor([out_tokens_residual], dtype=torch.int64, device=device)
        prosody_token_output = torch.tensor([out_tokens_prosody], dtype=torch.int64, device=device)

        return semantic_token_output, residual_token_output, prosody_token_output
