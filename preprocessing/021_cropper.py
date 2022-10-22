#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Copyright 2018 Luca Clissa, Marco Dalla, Roberto Morelli
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

"""
Created on Wed Jan  9 19:45:22 2019

@author: Roberto Morelli
"""


import glob
import sys
import numpy as np
from tqdm import tqdm
from subprocess import check_output
import matplotlib.pyplot as plt
from importlib import import_module
sys.path.append('..')
from utils import read_image_masks, read_masks, read_images
from config import *

LoadImagesForCrop = str(train_val_images).replace('preprocessing/', '') + '/'
LoadMasksForCrop = str(train_val_masks).replace('preprocessing/', '') + '/'

SaveCropImages = str(cropped_train_val_images).replace('preprocessing/', '') + '/'
SaveCropMasks = str(cropped_train_val_masks).replace('preprocessing/', '') + '/'


IMG_WIDTH = 1600
IMG_HEIGTH = 1200

XCropSize = 512
YCropSize = 512

XCropCoord = 400
YCropCoord = 400

XCropNum = int(IMG_WIDTH/XCropCoord)
YCropNum = int(IMG_HEIGTH/YCropCoord)

NumCropped = int(IMG_WIDTH/XCropCoord * IMG_HEIGTH/YCropCoord)

YShift = YCropSize - YCropCoord
XShift = XCropSize - XCropCoord

x_coord = [XCropCoord*i for i in range(0, XCropNum+1)]
y_coord = [YCropCoord*i for i in range(0, YCropNum+1)]

def cropper(image, mask):
        
    CroppedImgs = np.zeros((NumCropped, XCropSize, YCropSize, 3), np.uint8)
    CroppedMasks = np.zeros((NumCropped, XCropSize, YCropSize, 3), np.uint8)
    idx = 0
    
    for i in range(0, len(x_coord)-1):
        for j in range(0, len(y_coord)-1):
            
            if (i == 0) & (j == 0):
                CroppedImgs[idx] = image[y_coord[j]:y_coord[j+1] + YShift, x_coord[i]:x_coord[i+1] + XShift]
                CroppedMasks[idx] = mask[y_coord[j]:y_coord[j+1] + YShift, x_coord[i]:x_coord[i+1] + XShift]
                idx +=1 

            if (i == 0) & (j != 0):
                CroppedImgs[idx] = image[y_coord[j] - YShift : y_coord[j+1], x_coord[i]:x_coord[i+1] + XShift]
                CroppedMasks[idx] = mask[y_coord[j] - YShift : y_coord[j+1], x_coord[i]:x_coord[i+1] + XShift]
                idx +=1 
                
            if (i != 0) &  (j == 0):
                CroppedImgs[idx] = image[y_coord[j]:y_coord[j+1] + YShift, x_coord[i] - XShift :x_coord[i+1]]
                CroppedMasks[idx] = mask[y_coord[j]:y_coord[j+1] + YShift, x_coord[i] - XShift :x_coord[i+1]]
                idx +=1 
                
            if (i != 0) &  (j != 0):
                CroppedImgs[idx] = image[y_coord[j] - YShift : y_coord[j+1], x_coord[i] - XShift :x_coord[i+1]]
                CroppedMasks[idx] = mask[y_coord[j] - YShift : y_coord[j+1], x_coord[i] - XShift :x_coord[i+1]]
                idx +=1                             
            
    return CroppedImgs, CroppedMasks


def make_cropper():


    images_names = check_output(["ls", LoadImagesForCrop]).decode("utf8").split()

    for ix, name in tqdm(enumerate(images_names),total=len(images_names)):
        image, mask = read_image_masks(name, LoadImagesForCrop, LoadMasksForCrop)
        CroppedImages, CroppedMasks = cropper(image, mask)

        for i in range(0,NumCropped):

            crop_imgs_dir = os.path.join(SaveCropImages,name.split('.')[0]+'_{}'.format(i)+ '.' + name.split('.')[1])
            crop_masks_dir = os.path.join(SaveCropMasks,name.split('.')[0]+'_{}'.format(i)+ '.' +name.split('.')[1])

            plt.imsave(fname= crop_imgs_dir, arr = CroppedImages[i])
            plt.imsave(fname= crop_masks_dir,arr = CroppedMasks[i])

    return

    

if __name__ == "__main__":
    make_cropper()


