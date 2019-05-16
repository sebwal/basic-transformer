from model.constants import ATTENTION_CONST

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, masked=False):
        super(MultiHeadAttention, self).__init__()

        self.masked = masked
        self.attentionHeads = nn.ModuleList([SingleHeadAttention(masked) for _ in range(n_heads)])
        self.linear = nn.Linear(ATTENTION_CONST['mh_concat_width'], ATTENTION_CONST['mh_output_width'])

    def forward(self, inputs, encoderOutput=None):
        x = []
        for head in self.attentionHeads:
            sh_attention = head(inputs, encoderOutput=encoderOutput) 
            x.append(sh_attention)
        x = torch.cat(x, 1) # concatinate all single head attention outputs
        x = self.linear(x) # matmul with weight matrix (linear layer) to get 10x64 shape
        return x

class SingleHeadAttention(nn.Module):
    def __init__(self, masked):
        super(SingleHeadAttention, self).__init__()
        self.masked = masked
        self.linear1 = nn.Linear(ATTENTION_CONST['sh_linear1_input'], ATTENTION_CONST['sh_linear1_output'])
        self.linear2 = nn.Linear(ATTENTION_CONST['sh_linear2_input'], ATTENTION_CONST['sh_linear2_output'])
        self.linear3 = nn.Linear(ATTENTION_CONST['sh_linear3_input'], ATTENTION_CONST['sh_linear3_output'])
        self.scale = nn.Parameter(torch.FloatTensor([ATTENTION_CONST['sh_scale_factor']]))
        self.softmax = nn.Softmax()

    def forward(self, inputs, encoderOutput=None):        
        q = self.linear1(inputs)
        k = self.linear2(inputs) if encoderOutput == None else self.linear2(encoderOutput)
        v = self.linear3(inputs) if encoderOutput == None else self.linear3(encoderOutput)
        x = torch.matmul(q, k.permute(1, 0)) 
        x = x * self.scale
        # if self.masked:
        #     # TODO "future positions" have to be set to -inf. this is for the decoder to only allow self attention to consider earlier positions.
        x = self.softmax(x) 
        x = torch.matmul(x, v)
        return x


