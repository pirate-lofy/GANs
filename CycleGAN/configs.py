import torch
import numpy as np

device='cuda' if torch.cuda.is_available() else 'cpu'
BS=12
LR=2e-4
epochs=100
CYC_LAM=10
ID_LAM=0