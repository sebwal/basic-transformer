from model.multiheadattention import MultiHeadAttention
from model.feedforward import FeedForward
from model.constants import DECODER_CONST

import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, n_layers, n_attention_heads):
        super(Decoder, self).__init__()
        
        self.decoderLayers = nn.ModuleList([DecoderLayer(n_attention_heads) for _ in range(n_layers)])

    def forward(self, inputs):
        x = inputs
        for layer in self.decoderLayers:
            x = layer(x, inputs) # shape_in = , shape_out = 
        return x

class DecoderLayer(nn.Module):
    def __init__(self, n_attention_heads):
        super(DecoderLayer, self).__init__()

        self.mhattention_masked = MultiHeadAttention(n_attention_heads, masked=True)
        self.mhattention = MultiHeadAttention(n_attention_heads)
        self.feedforward = FeedForward(DECODER_CONST['ff1'], DECODER_CONST['ff2'])
        self.norm1 = nn.LayerNorm(DECODER_CONST['norm1_size'])
        self.norm2 = nn.LayerNorm(DECODER_CONST['norm2_size'])

    def forward(self, inputs, encoderOutput):
        x = inputs
        z = x
        print('dec layer mhattention_masked in', x.shape)
        x = self.mhattention_masked(x) # shape_in = , shape_out = 
        print('dec layer mhattention_masked out / norm1 in', x.shape)
        x = z + x
        x = self.norm1(x)
        print('dec layer norm1 out / mhattention in', x.shape)
        z = x
        x = self.mhattention(x, encoderOutput) # shape_in = , shape_out = 
        print('dec layer mhattention out / norm2 in', x.shape)
        x = z + x
        x = self.norm2(x)
        print('dec layer norm2 out / ff in', x.shape)
        z = x
        x = self.feedforward(x) # shape_in = , shape_out = 
        print('dec layer ff out / norm3 in', x.shape)
        x = z + x
        x = self.norm3(x)
        print('dec layer norm3 out', x.shape)
        return x

