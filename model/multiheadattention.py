from model.constants import ATTENTION_CONST

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, masked=False):
        super(MultiHeadAttention, self).__init__()

        self.masked = masked
        self.attentionHeads = nn.ModuleList([SingleHeadAttention(masked) for _ in range(n_heads)])
        self.linear1 = nn.Linear(ATTENTION_CONST['mh_concat_width'], ATTENTION_CONST['mh_output_width'])
        self.linear2 = nn.Linear(ATTENTION_CONST['mh_linear2_input'], ATTENTION_CONST['mh_linear2_output'])

    def forward(self, inputs, encoderOutput=None):
        x = []
        print('attention head in', inputs.shape)
        for head in self.attentionHeads: x.append(head(inputs, encoderOutput=encoderOutput)) # shape_in = , shape_out = 
        print('attention head out / concat in', x)
        x = self.concat(x)
        print('concat out / linear2 in', x[0].shape)        
        x = self.linear2(x) # shape_in = , shape_out = 
        return x

    def concat(self, matrices):
        return self.linear1(torch.cat(matrices, 1)) # shape_in = , shape_out = 

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
        q = self.linear1(inputs) # shape_in = , shape_out = 
        #TODO very unsure about this next part
        k = self.linear2(inputs) if encoderOutput == None else self.linear2(encoderOutput) # shape_in = , shape_out = 
        v = self.linear3(inputs) if encoderOutput == None else self.linear3(encoderOutput) # shape_in = , shape_out = 
        q = q.squeeze()
        k = k.squeeze()
        v = v.squeeze()
        x = torch.dot(q, k) # shape_in = , shape_out = 
        x = x * self.scale # shape_in = , shape_out = 
        # if self.masked:
        #     # TODO "future positions" have to be set to -inf. this is for the decoder to only allow self attention to consider earlier positions.
        x = self.softmax(x) # shape_in = , shape_out = 
        x = torch.dot(x, v) # shape_in = , shape_out = 
        return x


