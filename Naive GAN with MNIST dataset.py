import numpy as np
import cv2 as cv
from PIL import Image

import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.optim import Adam
from torch_snippets import *
from torchvision.utils import make_grid

device='cuda' if torch.cuda.is_available() else 'cpu'

def load_data():
    '''
        loads mnist dataset and apply transformation to ti
    '''
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,),std=(0.5,))
    ])
    B=128
    tr_data=MNIST('data/mnist',train=True,download=True,transform=transform)
    tr_loader=DataLoader(tr_data,shuffle=True,batch_size=B,drop_last=True)
    
    return tr_loader


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(28*28,1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(1024,512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.model(x)
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    def forward(self, x): return self.model(x)


def noise(size:int)->torch.Tensor:
    '''generated normally distributed random tensor'''
    return torch.randn(size,100).to(device)



def  disc_train_step(model:nn.Module,real_data:torch.Tensor,
                    fake_data:torch.Tensor,loss_fn,optim)->float:
    '''
        training the discriminator for one step
        by training the model on the fake and real data to
        learn how to predict 1 for real and 0 for fake
    '''
    # preparation
    optim.zero_grad()

    pred_real=model(real_data)
    real_loss=loss_fn(pred_real,torch.ones(len(real_data),1).to(device))  # we expect the discriminator to predict 1 for all real images
    real_loss.backward()  # computer gradients based on the real error

    pred_fake=model(fake_data)
    fake_loss=loss_fn(pred_fake,torch.zeros(len(fake_data),1).to(device))  # we expect the discriminator to predict 0 for all fake images
    fake_loss.backward()

    optim.step()

    return real_loss.item()+fake_loss.item()



def gen_train_step(model:nn.Module,fake_data:torch.Tensor,
                   real_data:torch.Tensor,loss_fn,optim)->float:
    '''
        training the generator for one step
        by passing the fake data into the discriminator 
        and train the generator to generate more realistic images
        to make the discriminator to predict them as 1
    '''
    optim.zero_grad()
    
    pred=model(fake_data)
    loss=loss_fn(pred,torch.ones(len(real_data),1).to(device))  # need the model to  predict 1 for fake data
    loss.backward()
    optim.step()
    return loss.item()



if __name__=='__main__':
    data=load_data()

    disc=Discriminator().to(device)
    gen=Generator().to(device)

    g_optim=Adam(gen.parameters(),lr=0.0002)
    d_optim=Adam(disc.parameters(),lr=0.0002)
    loss_fn=nn.BCELoss()

    epochs=3
    log=Report(epochs)

    for epoch in range(epochs):
        N=len(data)
        for i,(real_data,_) in enumerate(data):
            real_data=real_data.view(len(real_data),-1).to(device)
            fake_data=gen(noise(len(real_data)))
            fake_data.detach()
            dics_loss=disc_train_step(disc,real_data,fake_data,loss_fn,d_optim)

            fake_data=gen(noise(len(real_data)))
            gen_loss=gen_train_step(disc,fake_data,real_data,loss_fn,g_optim)

            log.record((i+1)/N,gen_loss=gen_loss,disc_loss=dics_loss,end='\r')
        
        log.report_avgs(epoch+1)
    
    # log.plot_epochs(['gen_loss','dics_loss'])


    z = torch.randn(64, 100).to(device)
    sample_images = gen(z).data.cpu().view(64, 1, 28, 28)
    grid = make_grid(sample_images, nrow=8, normalize=True)
    show(grid.cpu().detach().permute(1,2,0), sz=5)
