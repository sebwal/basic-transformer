from model.transformer import Transformer

import torch 
import torch.nn as nn
import random
import numpy as np

def main():
    transformer = Transformer()
    inputs = []
    for _ in range(10): inputs.append(random.randint(0, 1000))
    outputs = transformer(torch.Tensor(inputs).long())
    print(outputs)

if __name__ == "__main__":
    main()