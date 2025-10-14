import os
import soundfile as sf
import torch
import math
from hydra.utils import instantiate
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from src.utils import load_wav, basic_cleaners
from time import time
from datetime import datetime
from src.models.HubertTokenizer import HubertModel


import argparse
parser = argparse.ArgumentParser(description='Inference for MSR-Codec.')
parser.add_argument('--llm_ckpt', default='ckpt/llm.pth', type=str)
parser.add_argument('--codec_ckpt', default='ckpt/mel_codec.pth', type=str)
parser.add_argument('--vocoder_ckpt', default='ckpt/vocoder.pth', type=str)
parser.add_argument('--output_dir', type=str, default='output_dir')
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--top_p', type=float, default=0.8)
parser.add_argument('--top_k', type=int, default=25)
parser.add_argument('--repetition_penalty', type=float, default=1)
parser.add_argument('--prompt_wav', type=str, required=True)
parser.add_argument('--prompt_text', type=str, required=True)
parser.add_argument('--gen_text', type=str, required=True)

class HubertSynthesizer(object):
    def __init__(
        self,
        llm_ckpt,   
        codec_ckpt,  
        vocoder_ckpt,
        use_cpu=False,
        temperature=1.0,
        top_p=0.7,
        top_k=15,
        repetition_penalty=1.0,
        **unused
    ):
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.top_k = top_k
        

        self.llm_ckpt = llm_ckpt
        self.codec_ckpt = codec_ckpt
        self.vocoder_ckpt = vocoder_ckpt
        self.device = torch.device("cuda") if (torch.cuda.is_available() and (not use_cpu)) else torch.device("cpu")

    def setup(self):

        print('##### build text2token ######')
        print(f"loading model from {self.llm_ckpt}")
        llm_config = self.llm_ckpt.replace('.pth', '_config.yaml')
        mel_codec_config = self.codec_ckpt.replace('.pth', '_config.yaml')
        vocoder_config = self.vocoder_ckpt.replace('.pth', '_config.yaml')
        
        # load mel_codec: token to mel
        mel_codec_config = OmegaConf.load(mel_codec_config)
        self.mel_codec = instantiate(mel_codec_config.generator)
        ckpt = torch.load(self.codec_ckpt, map_location="cpu")
        self.mel_codec.load_state_dict(ckpt)
        
        self.mel_codec = self.mel_codec.to(self.device).eval()
        transform = instantiate(mel_codec_config.transform)
        self.mel_codec_transform = transform.to(self.device).eval()
        
        self.timbre_encoder = instantiate(mel_codec_config.timbre_encoder).to(self.device)
        self.timbre_encoder.eval().to(self.device)

        # build semantic token predictor
        model_config = OmegaConf.load(llm_config)
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_pretrained_path)
        model_config.model.text_token_size = len(self.text_tokenizer)
        model_config.model.codebook_size = mel_codec_config.generator.codebook_size
        model_config.model.spk_embed_dim = mel_codec_config.generator.timbre_dim
        model_config.model.spk_embed_dim_norm = mel_codec_config.timbre_encoder.norm_emb if hasattr(mel_codec_config.timbre_encoder, "norm_emb") else False

        model = instantiate(model_config.model)
        state_dict = torch.load(self.llm_ckpt, map_location="cpu")
        model.load_state_dict(state_dict)
        self.text2token = model.to(self.device).eval()

        # load vocoder: mel to wav
        vocoder_config = OmegaConf.load(vocoder_config)
        self.vocoder = instantiate(vocoder_config.generator)
        ckpt = torch.load(self.vocoder_ckpt, map_location="cpu")
        self.vocoder.load_state_dict(ckpt)
        self.vocoder = self.vocoder.to(self.device).eval()
        self.vocoder_sample_rate = vocoder_config.transform.orig_sample_rate

        self.hubert_encoder = HubertModel(self.device)

    def get_spk_emb(self, wav):
        with torch.no_grad():
            ori_wav_len = wav.size(-1)
            seg_num = math.ceil(ori_wav_len / self.mel_codec.segment_len)
            timbre_global_feat = self.timbre_encoder(wav)
            timbre_global_feat = timbre_global_feat.unsqueeze(-1).repeat(1, 1, seg_num)
        return timbre_global_feat

    @torch.no_grad()
    def synthesize_one_sample(self, target_text, prompt_text, prompt_speech_16k, include_prompt=False):
        prompt_text = basic_cleaners(prompt_text)
        res = self.text_tokenizer(prompt_text, return_tensors="pt")
        prompt_text = res["input_ids"].to(self.device)

        target_text = basic_cleaners(target_text)
        res = self.text_tokenizer(target_text, return_tensors="pt")
        target_text = res["input_ids"].to(self.device)

        prompt_speech_16k = prompt_speech_16k.to(self.device)
        speech_token = self.hubert_encoder(prompt_speech_16k).view(1, -1).long()

        mel = self.mel_codec_transform(prompt_speech_16k)
        timbre_seq = self.get_spk_emb(prompt_speech_16k)
        timbre_seq = timbre_seq.detach()
        output = self.mel_codec.inference(mel, speech_token, spk_emb=timbre_seq.transpose(1, 2), rec_mel=False)
        res_token, prosody_token = output[2],  output[3]

        res_token = res_token.detach().squeeze(1)
        prosody_token = prosody_token.detach().squeeze(1)

        text_len = target_text.size(-1)
        prompt_text_len = prompt_text.size(-1)

        if res_token.size(-1) > speech_token.size(-1):
            res_token = res_token[:, :speech_token.size(-1)]

        max_token_text_ratio = speech_token.size(-1) // 2 / text_len * 3
        min_token_text_ratio = max(speech_token.size(-1) // 2 / text_len, 0.8)
        
        semantic_token_output, \
        residual_token_output, \
        prosody_token_output, \
            = self.text2token.inference(
            text=target_text.to(self.device),
            text_len=torch.LongTensor([text_len]).reshape(-1).to(self.device),
            prompt_text=prompt_text.to(self.device),
            prompt_text_len=torch.LongTensor([prompt_text_len]).reshape(-1).to(self.device),
            prompt_semantic_token=speech_token,
            prompt_residual_token=res_token,
            prompt_prosody_token=prosody_token,
            prompt_timbre_seq=timbre_seq,
            max_token_text_ratio=max_token_text_ratio,
            min_token_text_ratio=min_token_text_ratio,
            use_global_embed=True,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )
        # print(semantic_token_output.size(), residual_token_output.size(), prosody_token_output.size(), timbre_output.size())
        semantic_token_pred = torch.cat([speech_token, semantic_token_output], dim=-1)
        residual_pred = torch.cat([res_token, residual_token_output], dim=-1)
        prosody_pred = torch.cat([prosody_token, prosody_token_output], dim=-1)
        
        # timbre_pred_ = self.get_spk_emb(prompt_speech_16k, True)
        timbre_pred = timbre_seq[:, :, 0].unsqueeze(-1) #.repeat(1, 1, timbre_pred.size(-1))

        mel_gen = self.mel_codec.generate(semantic_token_pred, residual_pred, prosody_pred, timbre_pred.transpose(1, 2))
        
        if not include_prompt:  
            mel_gen = mel_gen[:, :, mel.size(-1):]
        
        tts_speech = self.vocoder(mel_gen)
        return tts_speech


    @torch.no_grad()
    def synthesis(self, prompt_wav, prompt_text, gen_text, out_dir=None):
        if os.path.exists(out_dir) is False:
            os.makedirs(out_dir, exist_ok=True)
            
        start_time = time() 
        prompt_speech_16k, _ = load_wav(prompt_wav, sample_rate=16000, normalize=True)
        prompt_speech_16k = torch.from_numpy(prompt_speech_16k).unsqueeze(0).float()

        output = self.synthesize_one_sample(gen_text, prompt_text, prompt_speech_16k, include_prompt=False)

        pred_wave = output.cpu().numpy().reshape(-1)
        
        # 格式: YYYYMMDDHHMMSS
        time_str = datetime.now().strftime("%Y%m%d%H%M%S")
        out_n = os.path.join(out_dir, 'out_{}.wav'.format(time_str))
        sf.write(out_n, pred_wave, self.vocoder_sample_rate)
        end_time = time()
        print('Inference time:{}'.format(end_time - start_time))    


if __name__ == "__main__":
    args = parser.parse_args()
    model = HubertSynthesizer(llm_ckpt=args.llm_ckpt,  
                              codec_ckpt=args.codec_ckpt, 
                              vocoder_ckpt=args.vocoder_ckpt,
                              temperature=args.temperature,
                              top_p=args.top_p,
                              top_k=args.top_k,
                              repetition_penalty=args.repetition_penalty)
    model.setup()
    model.synthesis(args.prompt_wav, args.prompt_text, args.gen_text, args.output_dir)
