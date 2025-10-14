# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import torch
import torch.nn as nn

from src.modules.conformer.T5_conformer_decoder_layer import Conformer_T5Stack, T5Config, T5LayerNorm, T5Stack


class T5Conformer(torch.nn.Module):
    """Conformer encoder module.

    Args:
        config: T5Config,
        idim (int): Input dimension.
    """

    def __init__(
            self,
            config: T5Config,
            idim=512,
    ):
        """Construct an Encoder object."""
        super(T5Conformer, self).__init__()
        self.config = config
        print("T5Conformer is_decoder", config.is_decoder)

        if idim != config.d_model:
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, config.d_model),
                torch.nn.LayerNorm(config.d_model),
                torch.nn.Dropout(config.dropout_rate),
            )
            if config.is_decoder:
                self.dec_in_embed = torch.nn.Sequential(
                                        torch.nn.Linear(idim, config.d_model),
                                        torch.nn.LayerNorm(config.d_model),
                                        torch.nn.Dropout(config.dropout_rate),
                                    )
        else:
            self.embed = nn.Identity()
            self.dec_in_embed = nn.Identity()

        self.decoders = Conformer_T5Stack(config)
        self.after_norm = T5LayerNorm(config.d_model)

    def forward(self, 
                inputs_embeds, 
                attention_mask, 
                encoder_hidden_states=None, 
                encoder_attention_mask=None):
        """Encode input sequence.

        Args:
            xs (torch.Tensor): Input tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, time).

        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, time).

        """
        xs = self.embed(inputs_embeds)
        if self.config.is_decoder:
            encoder_hidden_states = self.dec_in_embed(encoder_hidden_states)

        xs = self.decoders(inputs_embeds=xs,
                           attention_mask=attention_mask,
                           encoder_hidden_states=encoder_hidden_states,
                           encoder_attention_mask=encoder_attention_mask)

        xs = self.after_norm(xs[0])
        return xs, attention_mask



class T5Stack_v2(torch.nn.Module):
    """Conformer encoder module.

    Args:
        config: T5Config,
        idim (int): Input dimension.
    """

    def __init__(
            self,
            config: T5Config,
            idim=512,
    ):
        """Construct an Encoder object."""
        super(T5Stack_v2, self).__init__()
        self.config = config
        if config.is_decoder is False:
            config.use_cache = False
        
        self.use_cache = config.use_cache

        if idim != config.d_model:
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, config.d_model),
                torch.nn.LayerNorm(config.d_model),
                torch.nn.Dropout(config.dropout_rate),
            )
            if config.is_decoder:
                self.dec_in_embed = torch.nn.Sequential(
                                        torch.nn.Linear(idim, config.d_model),
                                        torch.nn.LayerNorm(config.d_model),
                                        torch.nn.Dropout(config.dropout_rate),
                                    )
        else:
            self.embed = nn.Identity()
            self.dec_in_embed = nn.Identity()

        self.decoders = T5Stack(config)
        self.after_norm = T5LayerNorm(config.d_model)

    def forward(self, 
                inputs_embeds, 
                attention_mask, 
                encoder_hidden_states=None, 
                encoder_attention_mask=None):
        """Encode input sequence.

        Args:
            xs (torch.Tensor): Input tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, time).

        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, time).

        """

        xs = self.embed(inputs_embeds)

        if self.config.is_decoder:
            encoder_hidden_states = self.dec_in_embed(encoder_hidden_states)

        xs = self.decoders(inputs_embeds=xs,
                           attention_mask=attention_mask,
                           encoder_hidden_states=encoder_hidden_states,
                           encoder_attention_mask=encoder_attention_mask,
                           use_cache=self.use_cache)

        xs = self.after_norm(xs[0])
        return xs, attention_mask



if __name__ == '__main__':
    config_conv = T5Config(attention_dim=184, cnn_module_kernel=31, dropout_rate=0.2)
    config_fc = T5Config(attention_dim=184, linear_units=1536, positionwise_conv_kernel_size=3, dropout_rate=0.2)
    config = T5Config(d_model=184, d_kv=184, d_ff=1536, num_layers=2, dropout_rate=0.2, is_decoder=True,
                      positionwise_layer=config_fc,
                      convolution_layer=config_conv)

    x = torch.ones([10, 100, 512])
    # posi_emb = RelPositionalEncoding(184, dropout_rate=0.2)
    # x, emb = posi_emb(x)
    # print(x.size(), emb.size())
    y = torch.ones([10, 50, 184])

    net = T5Conformer(config)

    out = net(x, None, y, None)
    print(out[0].size())

    # print(out2[0].size())
