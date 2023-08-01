import cv2 as cv
import numpy as numpy
from torch import nn
from torch.utils.data import Dataset
from glob import glob
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

from configs import *

class GanDataset(Dataset):
  def __init__(self,path):
    super().__init__()

    self.list_imgs=glob(os.path.join(path,'*'))

    # augmentations
    self.trans=A.Compose([
      A.Resize(width=256,height=256),
      A.HorizontalFlip(p=0.5),
      A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],max_pixel_value=256),
      ToTensorV2()
    ],additional_targets={'image0':'image'})
    
    self.sample_trans=A.Compose([A.ColorJitter(p=0.1)])


  def __getitem__(self,indx):
    img=cv.imread(self.list_imgs[indx])
    w=img.shape[1]
    sample=img[:,:w//2]
    label=img[:,w//2:]
    
    sample=self.sample_trans(image=sample)['image']
    ret=self.trans(image=sample,image0=label)
    sample,label=ret['image'],ret['image0']
    return sample.to(device),label.to(device)

  def __len__(self):return len(self.list_imgs)


if __name__=='__main__':
    data=GanDataset('data/maps/maps/val')

    for g in range(5):
        i,j=data.__getitem__(g)
        i=i.permute([1,2,0]).numpy()[:,:,::-1]
        j=j.permute([1,2,0]).numpy()
        plt.imshow(i)
        plt.imshow(j)
        plt.show()