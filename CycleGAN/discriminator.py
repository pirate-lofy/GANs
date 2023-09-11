import torch
from torch import nn
from torchsummary import summary

'''
Ck denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2.
model=C64-C128-C256-C512
'''
class Block(nn.Module):
    def __init__(self,in_c,out_c,stride):
        super().__init__()

        self.model=nn.Sequential(
            nn.Conv2d(in_c,out_c,kernel_size=4,stride=stride,padding=1,padding_mode="reflect",bias=True),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2)
        )

    def forward(self,x):
        return self.model(x)
    
class Discriminator(nn.Module):
    def __init__(self,in_c=3,features=[64,128,256,512]):
        super().__init__()
        
        self.initial_layer=nn.Sequential(
            nn.Conv2d(in_c,features[0],stride=2,kernel_size=4,
                      padding=1,padding_mode='reflect',bias=True),
            nn.LeakyReLU(0.2)
        )
        layers=[]
        in_c=features[0]
        for feature in features[1:]:
            layers.append(Block(in_c,feature,stride=1 if feature==features[-1] else 2),)
            in_c=feature
        layers.append(nn.Conv2d(in_c,1,kernel_size=4,stride=1,
                                padding=1,padding_mode='reflect',bias=True))

        self.model=nn.Sequential(*layers)

    def forward(self,x):
        x=self.initial_layer(x)
        return self.model(x)
    
if __name__=='__main__':
    model=Discriminator()
    summary(model,(3,256,256))