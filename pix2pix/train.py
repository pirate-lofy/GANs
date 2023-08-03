from generator import Generator
from discriminator import Discriminator
from dataset import GanDataset
from configs import *

import torch
from torch.utils.data import DataLoader
from tqdm import trange,tqdm
from torch.optim import Adam
import cv2 as cv
import os
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import numpy as np
import argparse
import albumentations as A


def train_disc(disc,data,fake,bce,disc_optim,d_scaler):
    x,y=data

    with torch.cuda.amp.autocast():
        d_fake=disc(x,fake.detach())
        d_real=disc(x,y)
        fake_loss=bce(d_fake,torch.zeros_like(d_fake))
        real_loss=bce(d_real,torch.ones_like(d_real))
        loss=(fake_loss+real_loss)/2

    disc.zero_grad()
    d_scaler.scale(loss).backward()
    d_scaler.step(disc_optim)
    d_scaler.update()
    return loss.item()

def train_gen(disc,fake,x,y,bce,gen_optim,gen_scaler,L1):
    with torch.cuda.amp.autocast():
        d_fake=disc(x,fake.detach())
        fake_loss=bce(d_fake,torch.ones_like(d_fake))
        l1_loss=L1(fake,y)*l1_lambda
        loss=fake_loss+l1_loss

    gen_optim.zero_grad()
    gen_scaler.scale(loss).backward()
    gen_scaler.step(gen_optim)
    gen_scaler.update()
    return loss.item()

@torch.no_grad()
def validate_batch(data,gen,disc,bce,L1):
    x,y=data
    fake=gen(x)

    with torch.cuda.amp.autocast():
        d_fake=disc(x,fake.detach())
        fake_loss=bce(d_fake,torch.ones_like(d_fake))
        l1_loss=L1(fake,y)*l1_lambda
        disc_loss=fake_loss+l1_loss

        d_fake=disc(x,fake.detach())
        fake_loss=bce(d_fake,torch.ones_like(d_fake))
        l1_loss=L1(fake,y)*l1_lambda
        gen_loss=fake_loss+l1_loss

    return disc_loss.item(),gen_loss.item()


def save_sample(x,i):
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)
    transform=A.Compose([
            A.Normalize([-1,-1,-1],[1/0.5,1/0.5,1/0.5])
        ])
    for j,xx in enumerate(x):
        x=transform(x)
        fake=xx.permute([1,2,0]).detach().cpu().numpy()
        cv.imwrite(os.path.join(saving_path,f'{i}-{j}.jpg'),fake)


def main(args):
    writer=SummaryWriter()
    gen=Generator().to(device)
    disc=Discriminator().to(device)
    tr_loader=DataLoader(GanDataset(f'{args.data}/train'),batch_size=BS,shuffle=True)
    val_loader=DataLoader(GanDataset(f'{args.data}/val'),batch_size=8,shuffle=True)

    gen_optim=Adam(gen.parameters(),LR,betas=adam_betas)
    disc_optim=Adam(disc.parameters(),LR,betas=adam_betas)
    bce=torch.nn.BCEWithLogitsLoss()
    L1=torch.nn.L1Loss()
    gen_scaler=torch.cuda.amp.GradScaler()
    disc_scaler=torch.cuda.amp.GradScaler()

    for epoch in trange(epochs):
        vg_loss,vd_loss,tg_loss,td_loss=[],[],[],[]

        gen.train()
        disc.train()
        for ix,data in enumerate(tr_loader):
            x,y=data
            fake=gen(x)
            disc_loss=train_disc(disc,data,fake,bce,disc_optim,disc_scaler)
            gen_loss=train_gen(disc,fake,x,y,bce,gen_optim,gen_scaler,L1)
            td_loss.append(disc_loss)
            tg_loss.append(gen_loss)
        
        writer.add_scalar('training generator loss',np.mean(td_loss),epoch)
        writer.add_scalar('training disc loss',np.mean(td_loss),epoch)
            
        gen.eval()
        disc.eval()
        for ix,data in enumerate(val_loader):
            disc_loss,gen_loss=validate_batch(data,gen,disc,bce,L1)
            vd_loss.append(disc_loss)
            vg_loss.append(gen_loss)
        writer.add_scalar('validation generator loss',np.mean(td_loss),epoch)
        writer.add_scalar('calidation disc loss',np.mean(td_loss),epoch)

        if epoch%5==0:
            with torch.no_grad():
                x,y=next(iter(val_loader))
                fake_data=gen()
                real=make_grid(x,normalize=True)
                fake=make_grid(fake_data,normalize=True)
                writer.add_image('real',real,epoch)
                writer.add_image('fake',fake,epoch)

                save_sample(fake_data,epoch)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="A simple script that greets the user.")

    # Add arguments to the parser
    parser.add_argument("--data", type=str, help="path to the root of dataset")

    # Parse the command-line arguments
    args = parser.parse_args()
    main(args)
