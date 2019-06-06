from model.transformer import Transformer

import torch 
import torch.nn as nn
import random
import numpy as np

EPOCHS = 200

def main():
    transformer = Transformer()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0002)
    loss = torch.nn.BCELoss()

    real_sample = torch.Tensor([
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0, 0,0],
            [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0, 0,0],
            [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0, 0,0],
            [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0, 0,0],
            [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0, 0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0, 0, 0,0]
    ])
    for _ in range(1000):
        sample = transformer()        
        # target = torch.ones(sample.shape[0], sample.shape[1])
        error = loss(sample, real_sample)
        if _ % 10 == 0: print(error, sample)
        error.backward()
        optimizer.step()
        if(error > 8): break
    print(sample)

if __name__ == "__main__":
    main()