from model.encoder import Encoder
from model.decoder import Decoder
from model.constants import TRANS_CONST

import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, n_layers=TRANS_CONST['n_attention_layers'], n_attention_heads=TRANS_CONST['n_attention_heads']):
        super(Transformer, self).__init__()

        self.encoder = Encoder(n_layers, n_attention_heads)
        self.decoder = Decoder(n_layers, n_attention_heads)
        self.embedding = nn.Embedding(TRANS_CONST['embedding_dic_size'], TRANS_CONST['embedded_vec_size'])
        # self.posEncoding = #TODO
        self.linear = nn.Linear(TRANS_CONST['linear_input'], TRANS_CONST['linear_output'])
        self.softmax = nn.Softmax()

    def forward(self, inputs):
        x = self.doEmbedding(inputs) # shape_in = 1 x n_words, shape_out = n_words x length_embedding_vector
        # x = self.posEncoding(x) # no change in shape
        x = self.encoder(x) # shape_in = n_words x 512 (--> length of embedding), shape_out = 
        print('encoder out / decoder in', x.shape)
        x = self.decoder(x) # shape_in = , shape_out = 
        print('decoder out / trans linear in', x.shape)
        x = self.linear(x) # shape_in = , shape_out = 
        print('trans linear out / softmax in', x.shape)
        x = self.softmax(x) # shape_in = , shape_out = 
        print('trans softmax out', x.shape)
        return x
    
    def doEmbedding(self, inputs):
        x = []
        for word in inputs:
            # get every word as embedding vector from the embedding matrix
            embedded = self.embedding(word)
            # append unsqueezed so they're rows, not columns 
            x.append(embedded.unsqueeze(0))
        # concat all embedded tensors to one 2D tensor containing all embedded inputs
        x = torch.cat(x)
        return x