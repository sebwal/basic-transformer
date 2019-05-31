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

    for _ in range(1000):
        sample = transformer()
        target = torch.ones(sample.shape[0], sample.shape[1])
        error = loss(sample, target)
        if _ % 10 == 0: print(error)
        error.backward()
        optimizer.step()
        if(error > 8): break
    print(sample)

if __name__ == "__main__":
    main()