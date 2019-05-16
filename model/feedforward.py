import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, dim_model, dim_ff, dropout=0.1):
        #TODO
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(dim_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = inputs
        x = self.linear1(x) # shape_in = , shape_out = 
        x = nn.functional.relu(x) # shape_in = , shape_out = 
        x = self.dropout(x) # shape_in = , shape_out = 
        x = self.linear2(x) # shape_in = , shape_out = 
        return x
