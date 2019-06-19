from model.constants import FEEDFORWARD_CONST

import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, dim_model, dim_ff, dropout=FEEDFORWARD_CONST['dropout']):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(dim_model, dim_ff)
        # self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, dim_model)

    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        x = nn.functional.relu(x) 
        # x = self.dropout(x) 
        x = self.linear2(x) 
        return x
