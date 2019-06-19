from model.constants import ATTENTION_CONST

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.linear = nn.Linear(ATTENTION_CONST['mh_concat_width'], ATTENTION_CONST['mh_output_width'])
        self.wQ = nn.Linear(ATTENTION_CONST['sh_linear1_input'], ATTENTION_CONST['sh_linear1_output'])
        self.wK = nn.Linear(ATTENTION_CONST['sh_linear2_input'], ATTENTION_CONST['sh_linear2_output'])
        self.wV = nn.Linear(ATTENTION_CONST['sh_linear3_input'], ATTENTION_CONST['sh_linear3_output'])

    def forward(self, q, k, v):
        q = self.wQ(q)
        k = self.wK(k)
        v = self.wV(v)

        # split heads - I think they do this instead of a loop
        x, attention_weights = self.applyHeads(q, k, v)
        # transpose ?
        # reshape ?
        x = self.linear(x)
        return x, attention_weights

    def applyHeads(self, q, k, v, mask=None):
        x = torch.matmul(q, k.permute(1, 0)) 
        # scale x
        # add mask
        attention_weights = nn.Softmax(dim=-1)(x)
        x = torch.matmul(attention_weights, v)
        return x, attention_weights

