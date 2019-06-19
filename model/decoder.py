from model.multiheadattention import MultiHeadAttention
from model.feedforward import FeedForward
from model.constants import DECODER_CONST

import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, n_layers, n_attention_heads):
        super(Decoder, self).__init__()
        
        #self.embedding
        #self.pos_encoding
        #self.dropout
        self.decoderLayers = nn.ModuleList([DecoderLayer(n_attention_heads) for _ in range(n_layers)])

    def forward(self, inputs, encoderKV):
        x = inputs
        #x = self.embedding(x)
        #x = x * root of dim, whatever that is exactly
        #x = self.pos_encoding(x)
        #x = self.dropout(x)

        attention_weights = []
        for layer in self.decoderLayers:
            x, att1, att2 = layer(x, encoderKV) 
            attention_weights.append([att1, att2])

        return x, attention_weights

class DecoderLayer(nn.Module):
    def __init__(self, n_attention_heads):
        super(DecoderLayer, self).__init__()

        self.mhattention1 = MultiHeadAttention(n_attention_heads)
        self.mhattention2 = MultiHeadAttention(n_attention_heads)
        self.norm1 = nn.LayerNorm(DECODER_CONST['norm1_size'])
        self.norm2 = nn.LayerNorm(DECODER_CONST['norm2_size'])
        self.norm3 = nn.LayerNorm(DECODER_CONST['norm3_size'])
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()
        self.dropout3 = nn.Dropout()
        self.feedforward = FeedForward(DECODER_CONST['ff1'], DECODER_CONST['ff2'])

    def forward(self, inputs, encoderKV):
        x = inputs
        z = x
        x, att1 = self.mhattention1(x, x, x)
        x = self.dropout1(x)
        x = z + x        
        x = self.norm1(x)
        z = x
        x, att2 = self.mhattention2(x, encoderKV, encoderKV) 
        x = self.dropout2(x)
        x = z + x
        x = self.norm2(x)
        z = x
        x = self.feedforward(x)
        x = self.dropout3(x)
        x = z + x
        x = self.norm3(x)
        return x, att1, att2

