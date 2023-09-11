import torch
from torch import nn
from torchsummary import summary

'''
Let c7s1-k denote a 7×7 Convolution-InstanceNormReLU layer with k filters and stride 1
dk denotes a 3 × 3 Convolution-InstanceNorm-ReLU layer with k filters and stride 2
Rk denotes a residual block that contains two 3 × 3 convolutional layers with the same number of filters on both layer. 
uk denotes a 3 × 3 fractional-strided-Convolution-InstanceNorm-ReLU layer with k filters and stride 1/2

model6= c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3
model9= c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3
'''

# for c7s1,dk,uk
class CNNBlock(nn.Module):
    def __init__(self,in_c,out_c,k,s,down=True,act=True,**kwargs):
        super().__init__()
        
        self.model=nn.Sequential(
            nn.Conv2d(in_c,out_c,kernel_size=k,stride=s,padding=1,padding_mode='reflect',bias=True,**kwargs)
            if down else
            nn.ConvTranspose2d(in_c,out_c,kernel_size=3,stride=2,bias=True,**kwargs),
            nn.InstanceNorm2d(out_c),
            nn.ReLU() if act else nn.Identity()
        )

    def forward(self,x):
        return self.model(x)
    

class ResBlock(nn.Module):
    def __init__(self,c):
        super().__init__()

        self.model=nn.Sequential(
            CNNBlock(c,c,3,1),
            CNNBlock(c,c,3,1,act=False)
        )
    def forward(self,x):
        return x+self.model(x)


class Generator(nn.Module):
    def __init__(self,in_c=3,down_fs=[64,128,256],n_res=9,up_fs=[128,64]):
        super().__init__()

        self.initial=nn.Sequential(
            nn.Conv2d(in_c,down_fs[0],kernel_size=7,stride=1,padding=3,padding_mode='reflect',bias=True),
            nn.InstanceNorm2d(down_fs[0]),
            nn.ReLU()
        )

        layers=[]
        in_c=down_fs[0]
        for feature in down_fs[1:]:
            layers.append(CNNBlock(in_c,feature,3,2))
            in_c=feature
        
        for _ in range(n_res):
            layers.append(ResBlock(in_c))

        for feature in up_fs:
            layers.append(CNNBlock(in_c,feature,3,2,False,output_padding=1,padding=1))
            in_c=feature
         
        layers.append(nn.Conv2d(in_c,3,7,stride=1,padding=3))

        self.model=nn.Sequential(*layers)

    def forward(self,x):
        x=self.initial(x)
        return torch.tanh(self.model(x))
    


if __name__=='__main__':
    model=Generator()
    summary(model,(3,256,256))