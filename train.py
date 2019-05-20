from model.transformer import Transformer

import torch 
import torch.nn as nn
import random
import numpy as np

EPOCHS = 200

def main():
    transformer = Transformer()
    sample = transformer()
    print(sample)

if __name__ == "__main__":
    main()