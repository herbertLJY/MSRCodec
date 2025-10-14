#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder self-attention layer definition."""

import torch

from torch import nn

from transformers import T5Config, T5PreTrainedModel
from transformers.models.t5.modeling_t5 import T5Stack, T5Block, T5LayerNorm

from src.modules.conformer.layers_base import ConvolutionModule, MultiLayeredConv1d, Swish, RelPositionalEncoding
from src.modules.common import sequence_mask


class Conformer_T5Block(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)

    """

    def __init__(
            self,
            config: T5Config,
            has_relative_attention_bias=True,

    ):
        """Construct an EncoderLayer object."""
        super(Conformer_T5Block, self).__init__()
        if config.is_decoder is False:
            config.use_cache = False
        
        self.use_cache = config.use_cache
        print("Conformer_T5Block is_decoder", config.is_decoder)

        self.positionwise_layer1 = MultiLayeredConv1d(config.d_model,
                                                      config.positionwise_layer.linear_units,
                                                      config.positionwise_layer.positionwise_conv_kernel_size,
                                                      config.positionwise_layer.dropout_rate)

        self.positionwise_layer2 = MultiLayeredConv1d(config.d_model,
                                                      config.positionwise_layer.linear_units,
                                                      config.positionwise_layer.positionwise_conv_kernel_size,
                                                      config.positionwise_layer.dropout_rate)

        # convolution module definition
        convolution_layer = ConvolutionModule(config.d_model,
                                              config.convolution_layer.cnn_module_kernel,
                                              Swish())

        self.self_cross_att = T5Block(config, has_relative_attention_bias=has_relative_attention_bias)
        self.self_cross_att.layer[-1] = convolution_layer

        self.norm_ff_first = T5LayerNorm(config.d_model)
        self.norm_ff = T5LayerNorm(config.d_model)  # for the FNN module
        self.norm_final = T5LayerNorm(config.d_model)  # for the FNN module

        self.ff_scale = 0.5

        self.dropout = nn.Dropout(config.dropout_rate)

    # def forward(self, x_input, mask, memory, memory_mask, cache=None):
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_bias=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            encoder_decoder_position_bias=None,
            layer_head_mask=None,
            cross_attn_layer_head_mask=None,
            past_key_value=None,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
    ):
        """Compute encoded features.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        x = hidden_states

        residual = x
        x = self.norm_ff_first(x)
        x = residual + self.ff_scale * self.dropout(self.positionwise_layer1(x))
        use_cache = use_cache and self.use_cache

        outputs = self.self_cross_att(x,
                                      attention_mask=attention_mask,
                                      position_bias=position_bias,
                                      encoder_hidden_states=encoder_hidden_states,
                                      encoder_attention_mask=encoder_attention_mask,
                                      encoder_decoder_position_bias=encoder_decoder_position_bias,
                                      layer_head_mask=layer_head_mask,
                                      cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                                      past_key_value=past_key_value,
                                      use_cache=use_cache,
                                      output_attentions=output_attentions)

        x = outputs[0]
        # feed forward module
        residual = x
        x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.positionwise_layer2(x))

        hidden_states = self.norm_final(x)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,) + outputs[1:]

        return outputs


class Conformer_T5Stack(T5Stack):
    def __init__(self, config, embed_tokens=None):
        if config.is_decoder is False:
            config.use_cache = False
        super().__init__(config, embed_tokens)
        self.block = nn.ModuleList(
            [Conformer_T5Block(config, has_relative_attention_bias=True) for i in range(config.num_layers)]
        )


if __name__ == '__main__':
    # x = torch.rand([1, 2, 4, 5])
    # y = torch.randint(-1, 5, [1, 5])
    # yy = y.unsqueeze(1).unsqueeze(1)
    # out = x + y
    # print(x)
    # print(y)
    # print(x + y)
    # print(x + yy)

    config_conv = T5Config(cnn_module_kernel=31, dropout_rate=0.2)
    config_fc = T5Config(linear_units=1536, positionwise_conv_kernel_size=3, dropout_rate=0.2)
    config = T5Config(d_model=184, d_kv=184, d_ff=1536, num_layers=2, dropout_rate=0.2,
                      positionwise_layer=config_fc, is_decoder=False,
                      convolution_layer=config_conv,
                      use_cache=False, return_dict=False)

    x = torch.ones([2, 10, 184])
    # posi_emb = RelPositionalEncoding(184, dropout_rate=0.2)
    # x, emb = posi_emb(x)
    # print(x.size(), emb.size())
    y = torch.ones([2, 5, 184])
    mask = sequence_mask(torch.LongTensor([7, 5]), max_len=10)
    # print(mask)

    # infs = torch.FloatTensor([float("-inf"), 0]).unsqueeze(0).expand(x.size(0), -1)
    # mask = torch.gather(infs, -1, mask)
    #
    # print(mask)
    # mask = mask.unsqueeze(1).unsqueeze(1)

    # mask2 = sequence_mask(torch.LongTensor([7, 5]), max_len=10, dtype=torch.float).long()
    # print(mask2)

    # net = Conformer_T5Block(config, has_relative_attention_bias=True)
    #
    # out = net(hidden_states=x, attention_mask=mask, encoder_hidden_states=y)

    net2 = Conformer_T5Stack(config)

    out2 = net2(inputs_embeds=x) #, attention_mask=mask, encoder_hidden_states=y)

    print(out2)
    for x in out2:
        print(x.size())
    
    print(out2[0].size())
