import torch
import argparse
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from generator import Generator
from discriminator import Discriminator
from dataset import DATA
from configs import *


def main(args):
    gen_h=Generator().to(device)
    gen_z=Generator().to(device)
    disc_h=Discriminator().to(device)
    disc_z=Discriminator().to(device)

    gen_optim=Adam(
        list(gen_h.parameters())+list(gen_z.parameters()),
        lr=LR,
        betas=(0.5,0.99)
    )
    disc_optim=Adam(
        list(disc_h.parameters())+list(disc_z.parameters()),
        lr=LR,
        betas=(0.5,0.99)
    )

    L1=nn.L1Loss()
    MSE=nn.MSELoss()

    tr_data=DataLoader(
        DATA(args.A,args.B),
        shuffle=True,batch_size=BS,pin_memory=True)
    # ts_data=DataLoader(
    #     DATA('data/horse2zebra/testA','data/horse2zebra/testB'),
    #     shuffle=True,batch_size=BS,pin_memory=True)
    
    g_scalar=torch.cuda.amp.GradScaler()
    d_scalar=torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        disc_losses=[]
        gen_losses=[]

        for horse,zebra in tr_data:
            horse=horse.to(device)
            zebra=zebra.to(device)
            
            with torch.cuda.amp.autocast():
                # train H disc
                fake_h=gen_h(zebra)
                d_h_fake=disc_h(fake_h.detach())
                d_h_real=disc_h(horse)
                h_loss_real=MSE(d_h_real,torch.ones_like(d_h_real))
                h_loss_fake=MSE(d_h_fake,torch.zeros_like(d_h_fake))
                d_h_loss=h_loss_fake+h_loss_real
                
                # train Z disc
                fake_z=gen_z(horse)
                d_z_fake=disc_z(fake_z.detach())
                d_z_real=disc_z(zebra)
                z_loss_real=MSE(d_z_real,torch.ones_like(d_z_real))
                z_loss_fake=MSE(d_z_fake,torch.zeros_like(d_z_fake))
                d_z_loss=z_loss_fake+z_loss_real

                d_loss=(d_z_loss+d_h_loss)/2
                disc_losses.append(d_loss.item())

            disc_optim.zero_grad()
            d_scalar.scale(d_loss).backward()
            d_scalar.step(disc_optim)
            d_scalar.update()


            # train H,Z Gen
            with torch.cuda.amp.autocast():
                d_h_fake=disc_h(fake_h)
                d_z_fake=disc_z(fake_z)
                g_h_loss=MSE(d_h_fake,torch.ones_like(d_h_fake))
                g_z_loss=MSE(d_z_fake,torch.ones_like(d_z_fake))

                # cycle loss
                cycle_h=gen_h(fake_z)
                cycle_z=gen_z(fake_h)
                cycle_h_loss=L1(horse,cycle_h)
                cycle_z_loss=L1(zebra,cycle_z)

                # identity loss
                if ID_LAM:
                    identity_h=gen_h(fake_h)
                    identity_z=gen_z(fake_z)
                    identity_h_loss=L1(identity_h,horse)
                    identity_z_loss=L1(identity_z,zebra)

                g_loss=g_h_loss+g_z_loss+cycle_h_loss*CYC_LAM+cycle_z_loss*CYC_LAM#+identity_h_loss*ID_LAM+identity_z_loss*ID_LAM
                gen_losses.append(g_loss.item())

            gen_optim.zero_grad()
            g_scalar.scale(g_loss).backward()
            g_scalar.step(gen_optim)
            g_scalar.update()
        
        print(f'epoch:{epoch}, gen loss:{round(np.mean(gen_losses),4)} disc losses: {round(np.mean(disc_losses),4)}')
        if epoch%2:
            save_image(fake_h,f'samples/horses/{epoch}.jpg')
            save_image(fake_z,f'samples/zebras/{epoch}.jpg')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="A simple script that greets the user.")

    # Add arguments to the parser
    parser.add_argument("--A", type=str, help="path to the A subset of dataset")
    parser.add_argument("--B", type=str, help="path to the B subset of dataset")

    # Parse the command-line arguments
    args = parser.parse_args()
    main(args)