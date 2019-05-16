from model.multiheadattention import MultiHeadAttention
from model.feedforward import FeedForward
from model.constants import DECODER_CONST

import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, n_layers, n_attention_heads):
        super(Decoder, self).__init__()
        
        self.decoderLayers = nn.ModuleList([DecoderLayer(n_attention_heads) for _ in range(n_layers)])

    def forward(self, inputs, encoderKV):
        x = inputs
        for layer in self.decoderLayers:
            x = layer(x, encoderKV) 
        return x

class DecoderLayer(nn.Module):
    def __init__(self, n_attention_heads):
        super(DecoderLayer, self).__init__()

        self.mhattention_masked = MultiHeadAttention(n_attention_heads, masked=True)
        self.mhattention = MultiHeadAttention(n_attention_heads)
        self.feedforward = FeedForward(DECODER_CONST['ff1'], DECODER_CONST['ff2'])
        self.norm1 = nn.LayerNorm(DECODER_CONST['norm1_size'])
        self.norm2 = nn.LayerNorm(DECODER_CONST['norm2_size'])
        self.norm3 = nn.LayerNorm(DECODER_CONST['norm3_size'])

    def forward(self, inputs, encoderKV):
        x = inputs
        z = x
        x = self.mhattention_masked(x) #TODO masking not implemented
        x = z + x
        x = self.norm1(x)
        z = x
        x = self.mhattention(x, encoderKV=encoderKV) 
        x = z + x
        x = self.norm2(x)
        z = x
        x = self.feedforward(x)
        x = z + x
        x = self.norm3(x)
        return x

