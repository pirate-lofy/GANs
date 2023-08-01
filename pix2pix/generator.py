import torch
from torch import nn
from torchsummary import summary

class Block(nn.Module):
    def __init__(self,inp,out,down=True,act='relu',dropout=False):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(inp,out,4,2,1,bias=False,padding_mode='reflect')
            if down else
            nn.ConvTranspose2d(inp,out,4,2,1,bias=False),

            nn.BatchNorm2d(out),
            
            nn.ReLU() 
            if act=='relu' else 
            nn.LeakyReLU(0.2)
        )
        self.dropout=nn.Dropout(0.5) if dropout else None

    def forward(self,x):
        x=self.conv(x)
        return self.dropout(x) if self.dropout else x

class Generator(nn.Module):
    def __init__(self,in_c=3,out_c=64):
        super().__init__()

        self.initial_down=nn.Sequential(
            nn.Conv2d(in_c,out_c,4,2,1,bias=False,padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        self.down1=Block(out_c,out_c*2,True,'leaky',False)      # 64
        self.down2=Block(out_c*2,out_c*4,True,'leaky',False)    
        self.down3=Block(out_c*4,out_c*8,True,'leaky',False)
        self.down4=Block(out_c*8,out_c*8,True,'leaky',False)
        self.down5=Block(out_c*8,out_c*8,True,'leaky',False)
        self.down6=Block(out_c*8,out_c*8,True,'leaky',False)

        self.bottleneck=nn.Sequential(
            nn.Conv2d(out_c*8,out_c*8,4,2,1,padding_mode='reflect'),
            nn.ReLU()
        )

        self.up1=Block(out_c*8,out_c*8,False,'relu',True)
        self.up2=Block(out_c*8*2,out_c*8,False,'relu',True)
        self.up3=Block(out_c*8*2,out_c*8,False,'relu',True)
        self.up4=Block(out_c*8*2,out_c*8,False,'relu',False)
        self.up5=Block(out_c*8*2,out_c*4,False,'relu',False)
        self.up6=Block(out_c*4*2,out_c*2,False,'relu',False)
        self.up7=Block(out_c*2*2,out_c,False,'relu',False)

        self.final_up=nn.Sequential(
            nn.ConvTranspose2d(out_c*2,in_c,4,2,1),
            nn.Tanh()
        )


    def forward(self,x):
        d0= self.initial_down(x)
        d1=self.down1(d0)
        d2=self.down2(d1)
        d3=self.down3(d2)
        d4=self.down4(d3)
        d5=self.down5(d4)
        d6=self.down6(d5)
        bottle=self.bottleneck(d6)
        up1=self.up1(bottle)
        up2=self.up2(torch.cat([up1,d6],1))
        up3=self.up3(torch.cat([up2,d5],1))
        up4=self.up4(torch.cat([up3,d4],1))
        up5=self.up5(torch.cat([up4,d3],1))
        up6=self.up6(torch.cat([up5,d2],1))
        up7=self.up7(torch.cat([up6,d1],1))
        f=self.final_up(torch.cat([up7,d0],1))
        return f
    

if __name__=='__main__':
    model=Generator()
    summary(model,(3,256,256))