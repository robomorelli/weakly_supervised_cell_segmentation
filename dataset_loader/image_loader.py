import pathlib

import pandas as pd
import torch
from torchvision.transforms import transforms as T
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from config import *

class CellsLoader(Dataset):
    def __init__(self, images_path=None, masks_path=None, val_split=0.3, grayscale=False,
                 transform=None, ae=None, test=False, priority_list=[], exclude_RT=False, regression=False):
        try:
            self.imgs_path = images_path.as_posix() + '/'
            self.masks_path = masks_path.as_posix() + '/'
        except:
            self.imgs_path = images_path + '/'
            self.masks_path = masks_path + '/'
        self.val_split = val_split
        self.transform = transform
        self.ae = ae
        self.test = test
        self.priority_list = priority_list
        self.grayscale = grayscale
        self.regression = regression

        #if self.grayscale:
        #    self.transform_gray = transform.transforms.append(T.Resize((1040,1400)))

        self.imgs_list = os.listdir(self.imgs_path)
        self.masks_list = os.listdir(self.masks_path)

        if self.regression:
            labels_df = pd.read_csv(str(labels_csv).replace('preprocessing', ''))
            labels_df.drop_duplicates(subset=["img_name"]).set_index(['img_name'], inplace=True)
            self.counts = labels_df.loc[self.masks_list, :]['count'].values

        if exclude_RT:
            for n in self.imgs_list:
                if 'RT' in n:
                    self.imgs_list.remove(n)
                    self.masks_list.remove(n)
                    print('removing from test image', n)
            print(self.imgs_list)

    def __len__(self):
        return len(self.imgs_list)
    def __getitem__(self, idx):

        if self.ae == 'ae' and not self.test:
            img = cv2.imread(self.imgs_path + self.imgs_list[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform is not None:
                img = self.transform(img)
            return img.float(), img.float()
        else:
            img = cv2.imread(self.imgs_path + self.imgs_list[idx])
            if self.grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if not self.regression:
                mask = cv2.imread(self.masks_path + self.masks_list[idx])
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            else:
                mask = self.counts[idx]
            if self.transform is not None:
                mask = self.transform(mask)
                img = self.transform(img)
            return img.float(), mask.float()
