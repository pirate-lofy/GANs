import torch
from torch import nn
from torchsummary import summary

class Block(nn.Module):
    def __init__(self,in_c,out_c,strid):
        super().__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_c,out_c,4,strid,bias=False,padding_mode='reflect'),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2)
        )
    def forward(self,x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self,in_c=3,features=[64,128,256,512]):
        super().__init__()
        self.initial_layer=nn.Sequential(
            nn.Conv2d(in_c*2,features[0],4,2,padding=1,padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )

        layers=[]
        in_c=features[0]
        for feature in features[1:]:
            layers.append(Block(in_c,feature,strid=1 if feature==features[-1] else 2))
            in_c=feature

        layers.append(nn.Conv2d(
            in_c,1,4,1,1,padding_mode='reflect'
        ))
        self.model=nn.Sequential(*layers)

    def forward(self,x,y):
        # print(x.shape,y.shape)
        inp=torch.cat([x,y],dim=1)
        inp=self.initial_layer(inp)
        return self.model(inp)
    

if __name__=='__main__':
    # summary(Discriminator(),(3,256,256),(3,256,256))
    x=torch.randn([1,3,256,256])
    y=torch.randn([1,3,256,256])
    model=Discriminator()
    print(model(x,y).shape)