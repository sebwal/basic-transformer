# import numpy as np
# import torch
import torch.nn as nn
# import torch.nn.functional as F
# import math, copy, time
# from torch.autograd import Variable

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)