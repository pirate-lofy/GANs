'''
    The implemntation of the paper https://arxiv.org/pdf/1511.06434.pdf

    Implementation Notes according to the original paper
        1. Use strided convolution in the model archeticture instead of pooling
        2. Use batchnormalization in both generator and discriminator instead of 
            the input layer of the discriminator and the output layer of the generator
            to stablilize the training
        3. Use ReLU in all of the generator's layer but the last layer use Tanh
        4. Normalize images to the range od the Tansh function [-1,1]
        5. Remove the fully connected layers in deep models

    Troubleshooting:

        1. generator loss was getting larger while discriminator loss was near zero
            a LeakyReeLU layer was missing in the discriminator layer (do not no why)

        2. can not denormalize images comming out of the generator
            using the inverse of the normalization values and multiply by 255
'''


import os
import numpy as np
import torch
from glob import glob
from torch import nn
import cv2 as cv
from torch_snippets import *
from torch.optim import Adam
from torchvision import transforms
from torchsummary import summary
import torchvision.utils as vutils
from torch.utils.data import Dataset,DataLoader

device='cuda' if torch.cuda.is_available() else 'cpu'

def noise(size):
    '''generates random tensor of specific size'''
    return torch.randn(size,100,1,1).to(device)

def init(m):
    '''
        initializes the weights of all presented layers in the model
        the initialization values were introduced in the paper of DCGAN https://arxiv.org/pdf/1511.06434.pdf    
    '''
    classname=m.__class__.__name__
    if classname.find('Conv')!=-1:
        nn.init.normal_(m.weight.data,0.0,0.2)
    elif classname.find('BatchNorm')!=-1:
        nn.init.normal_(m.weight.data,1.0,0.2)
        nn.init.constant_(m.bias.data,0)

class Faces(Dataset):
    def __init__(self,males,females):
        super().__init__() 
        self.cascade=cv.CascadeClassifier(cv.data.haarcascades+'haarcascade_frontalface_default.xml') # to crop faces

        males=glob(os.path.join(males,'*'))
        females=glob(os.path.join(females,'*'))
        paths=np.array(males+females)
        self.paths=self.prepare_faces(paths,'data/male_female/faces')
        
        self.transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])


    def prepare_faces(self,paths:str,save_p:str)->list:
        '''iterate on all images and crop faces then save them'''
        if os.path.exists(save_p):
            return glob(os.path.join(save_p,'*'))
        os.mkdir(save_p)

        for i,p in enumerate(paths):
            img=cv.imread(p)
            gray=cv.imread(p,0)
            faces=self.cascade.detectMultiScale(gray,1.3,5)
            for j,(x,y,w,h) in enumerate(faces):
                p=os.path.join(save_p,f'{i}-{j}.jpg')
                cv.imwrite(p,img[x:x+w,y:y+h])
        return glob(os.path.join(save_p,'*'))

    def __getitem__(self,i):
        img=cv.imread(self.paths[i]) 
        img=self.transform(img)
        return img.to(device)

    def __len__(self):return len(self.paths)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # input shape [Batch * 100 * 1 * 1]
        self.model=nn.Sequential(
            nn.ConvTranspose2d(100,64*8,4,1,0,bias=False), # excluding bias as we are using batchnorm
            nn.BatchNorm2d(64*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(64*8,64*4,4,2,1,bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(64*4,64*2,4,2,1,bias=False),
            nn.BatchNorm2d(64*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(64*2,64,4,2,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64,3,4,2,1,bias=False),
            nn.Tanh()
        )
        self.apply(init)
    

    def forward(self,x):
        return self.model(x)
    

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # input shape [Batch * Channels * 64 * 64]
        self.model=nn.Sequential(
            nn.Conv2d(3,64,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(64,64*2,4,2,1,bias=False),
            nn.BatchNorm2d(64*2),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(64*2,64*4,4,2,1,bias=False),
            nn.BatchNorm2d(64*4),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(64*4,64*8,4,2,1,bias=False),
            nn.BatchNorm2d(64*8),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(64*8,1,4,1,0,bias=False),
            nn.Sigmoid()
        )
        self.apply(init)
    
    def forward(self,x):
        return self.model(x)
    

def gen_train_step(model:nn.Module,fake_data:torch.Tensor,
                   loss_fn,optim)->float:
    optim.zero_grad()
    pred_fake=model(fake_data).squeeze()
    loss=loss_fn(pred_fake,torch.ones(len(fake_data)).to(device))
    loss.backward()
    optim.step()
    return loss.item()

def disc_train_step(model:nn.Module,fake_data:torch.Tensor,
                    real_data:torch.Tensor,loss_fn,optim)->float:
    optim.zero_grad()
    pred_fake=model(fake_data).squeeze()
    
    fake_loss=loss_fn(pred_fake,torch.zeros(len(fake_data)).to(device))
    fake_loss.backward()

    pred_real=model(real_data).squeeze()
    real_loss=loss_fn(pred_real,torch.ones(len(real_data)).to(device))
    real_loss.backward()

    optim.step()
    return real_loss.item()+fake_loss.item()


if __name__=='__main__':
    gen=Generator().to(device)
    disc=Discriminator().to(device)
    dataloader=DataLoader(Faces('data/male_female/females','data/male_female/males'),
                          shuffle=True,batch_size=64)

    loss_fn=nn.BCELoss()
    g_optim=Adam(gen.parameters(),0.0002,betas=(0.5,0.999)) # to stabilize the training
    d_optim=Adam(disc.parameters(),0.0002,betas=(0.5,0.999))

    epochs=2
    log=Report(epochs)

    for epoch in range(epochs):
        N=len(dataloader)
        for i,real_data in enumerate(dataloader):
            fake_data=gen(noise(len(real_data)))
            fake_data.detach()
            d_loss=disc_train_step(disc,fake_data,real_data,
                                   loss_fn,d_optim)
            
            fake_data=gen(noise(len(real_data)))
            g_loss=gen_train_step(disc,fake_data,loss_fn,g_optim)
            log.record((i+1)/N,d_loss=d_loss,g_loss=g_loss,end='/t')
        log.report_avgs(epoch+1)


    # evaluation
    gen.eval()
    noise = torch.randn(1, 100, 1, 1, device=device)

    denormalize=transforms.Compose([
                transforms.Normalize([-1,-1,-1],[1/0.5,1/0.5,1/0.5])
            ])
    imgs = denormalize(gen(noise))
    img=imgs.detach().cpu().permute(0,2,3,1).numpy()[0]*255

    
        