import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from src.utils import load_wav
from src.modules.campplus.DTDNN import CAMPPlus

class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97):
        super(PreEmphasis, self).__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        # print(input.size(), 111)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class LogMelBank(torch.nn.Module):
    def __init__(self, out_dim=64, mean_nor=False, **unused):
        super(LogMelBank, self).__init__()
        SAMPING_RATE = 16000
        self.mean_nor = mean_nor
        win_length = 400  # args.fft_window_size
        hop_length = 160  # args.fft_window_stride
        print('Do STFT with window size:{}, stride:{}'.format(win_length, hop_length))

        self.pre = PreEmphasis()
        self.mel_bank = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPING_RATE, n_fft=512,
                                                             win_length=win_length, hop_length=hop_length,
                                                             window_fn=lambda x: torch.hann_window(x, periodic=False).pow(0.85), n_mels=out_dim)
        self.out_dim = out_dim

# torch.hann_window(window_size, periodic=False, device=device, dtype=dtype).pow(0.85)

    def forward_base(self, wav_input):
        wav_input = wav_input.unsqueeze(1)
        wav_input = self.pre(wav_input)
        wav_input = wav_input.squeeze(1)
        wav_out = self.mel_bank(wav_input)
        wav_out = torch.log(wav_out + 1e-6)
        return wav_out

    def forward(self, wav_input):
        wav_out = self.forward_base(wav_input)
        if self.mean_nor:
            wav_out = wav_out - torch.mean(wav_out, dim=-1, keepdim=True)
        wav_out = wav_out.transpose(1, 2)
        return wav_out


class CAMP_plus(nn.Module):
    def __init__(self, ckpt_path='ckpt/campplus_cn_en_common.pt', norm_emb=False, **unused):
        super(CAMP_plus, self).__init__()
        pretrained_model = ckpt_path
        pretrained_state = torch.load(pretrained_model, map_location='cpu')
        self.norm_emb = norm_emb

        # load model
        embedding_model = CAMPPlus(feat_dim=80, embedding_size=192) #.load_state_dict(pretrained_state)
        embedding_model.load_state_dict(pretrained_state)
        self.embedding_model = embedding_model
        self.embedding_model.eval()

        self.feature_extractor = LogMelBank(80, mean_nor=True)

    def forward(self, wav, mel=None, with_grad=False):
        if mel is None:
            feat = self.feature_extractor(wav)
        else:
            feat = mel
        if with_grad:
            embedding = self.embedding_model(feat)
        else:
            with torch.no_grad():
                embedding = self.embedding_model(feat).detach()
        if self.norm_emb:
            embedding = F.normalize(embedding, dim=-1)
        return embedding
