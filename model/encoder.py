from model.multiheadattention import MultiHeadAttention
from model.feedforward import FeedForward
from model.constants import ENCODER_CONST

import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, n_layer, n_attention_heads):
        super(Encoder, self).__init__()
        
        self.encoderLayers = nn.ModuleList([EncoderLayer(n_attention_heads) for _ in range(n_layer)])

    def forward(self, inputs):
        x = inputs
        for layer in self.encoderLayers:
            x = layer(x) # shape_in = , shape_out = 
        return x

class EncoderLayer(nn.Module):
    def __init__(self, n_attention_heads):
        super(EncoderLayer, self).__init__()

        self.mhattention = MultiHeadAttention(n_attention_heads)
        self.feedforward = FeedForward(ENCODER_CONST['ff1'], ENCODER_CONST['ff2'])
        self.norm1 = nn.LayerNorm(ENCODER_CONST['norm1_size'])
        self.norm2 = nn.LayerNorm(ENCODER_CONST['norm2_size'])

    def forward(self, inputs):
        x = inputs 
        z = x
        print('enc layer mhattention in', x.shape)
        x = self.mhattention(x) # shape_in = , shape_out = 
        print('enc layer mh attention out / norm1 in', x.shape)
        x = self.norm1(z + x) # shape_in = , shape_out = 
        print('enc layer norm1 out / ff in', x.shape)
        z = x
        x = self.feedforward(x) # shape_in = , shape_out = 
        print('enc layer ff out / norm2 in', x.shape)
        x = self.norm2(z + x) # shape_in = , shape_out = 
        print('enc layer norm2 out', x.shape)
        return x

