import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob
import cv2 as cv
from random import shuffle
from configs import *

class DATA(Dataset):
    def __init__(self,hosres_path,zerbras_path):
        super().__init__()

        self.horses=glob(hosres_path+'/*')
        shuffle(self.horses)
        self.zebras=glob(zerbras_path+'/*')
        shuffle(self.zebras)

        self.transform=A.Compose([
            A.Resize(256,256),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
            ToTensorV2()
        ],
        additional_targets={'image0':'image'}
        )

    def __len__(self):return min(len(self.horses),len(self.zebras))

    def __getitem__(self,indx):
        horse=cv.imread(self.horses[indx])
        zebra=cv.imread(self.zebras[indx])
        aug=self.transform(image=horse,image0=zebra)
        horse,zebra=aug['image'],aug['image0']

        return horse,zebra


if __name__=='__main__':
    data=DATA('/media/bignrz/Fast Data/study/modern computer vision with pytorch/my impl/ch12/GANs/data/horse2zebra/trainA',
              '/media/bignrz/Fast Data/study/modern computer vision with pytorch/my impl/ch12/GANs/data/horse2zebra/trainB')
    horse,zebra=data.__getitem__(0)