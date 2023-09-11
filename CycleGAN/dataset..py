import torch
from torch.utils.data import Dataset,DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob
import cv2 as cv
from random import shuffle

class DATA(Dataset):
    def __init__(self,hosres_path,zerbras_path):
        super().__init__()

        self.horses=shuffle(glob(hosres_path+'/*'))
        self.zebras=shuffle(glob(zerbras_path+'/*'))

        self.transform=A.Compose(
            A.Resize(256,256),
            A.HirzontalFlip(p=0.5)
            A.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
            ToTensorV2()
        )

    def __len__(self):return min(len(self.horses),len(self.zebras))

    def __getitem__(self,indx):
        horse=cv.imread(self.horses[indx])
        zebra=cv.imread(self.zebras[indx])

        if self.transform:
            aug=self.transform(image=horse,image0=zebra)
            hosre,zebra=aug['image'],aug['image0']

        return horse,zebra


