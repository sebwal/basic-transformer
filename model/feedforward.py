# import numpy as np
# import torch
import torch.nn as nn
# import torch.nn.functional as F
# import math, copy, time
# from torch.autograd import Variable

class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim_model, dim_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(dim_model, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, dim_model)

    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        x = nn.functional.relu(x) 
        x = self.dropout(x) 
        x = self.linear2(x) 
        return x
