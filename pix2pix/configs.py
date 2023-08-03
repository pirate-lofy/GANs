import torch 

LR=2e-4
epochs=30
l1_lambda=100
device='cuda' if torch.cuda.is_available() else 'cpu'
adam_betas=(0.5,0.999)
BS=128
saving_path='samples'