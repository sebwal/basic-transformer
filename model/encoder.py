from model.multiheadattention import MultiHeadAttention
from model.feedforward import FeedForward
from model.constants import ENCODER_CONST

import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, n_layer, n_attention_heads):
        super(Encoder, self).__init__()
        
        #self.embedding
        #self.pos_encoding
        #self.dropout
        self.encoderLayers = nn.ModuleList([EncoderLayer(n_attention_heads) for _ in range(n_layer)])

    def forward(self, inputs):
        x = inputs
        #x = self.embedding(x)
        #x = x * root of dim, whatever that is exactly
        #x = self.pos_encoding(x)
        #x = self.dropout(x)

        for layer in self.encoderLayers:
            x = layer(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, n_attention_heads):
        super(EncoderLayer, self).__init__()

        self.mhattention = MultiHeadAttention(n_attention_heads)
        self.norm1 = nn.LayerNorm(ENCODER_CONST['norm1_size'])
        self.norm2 = nn.LayerNorm(ENCODER_CONST['norm2_size'])
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()
        self.feedforward = FeedForward(ENCODER_CONST['ff1'], ENCODER_CONST['ff2'])

    def forward(self, inputs):
        x = inputs 
        z = x
        x = self.mhattention(x, x, x)
        x = self.dropout1(x)
        x = z + x
        x = self.norm1(x)
        z = x
        x = self.feedforward(x)
        x = self.dropout2(x)
        x = z + x
        x = self.norm2(x) 
        return x

