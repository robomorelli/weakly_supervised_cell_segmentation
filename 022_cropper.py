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
import imageio
import cv2
import random
from skimage import transform
import os
from tqdm import tqdm
from subprocess import check_output
import matplotlib.pyplot as plt
from importlib import import_module
reader = import_module('020_loader')

from config_script import *    

LoadImagesForCrop = TRAIN_VALID_OR_PATH # Image without Test
LoadLabelsForCrop = TRAIN_VALID_MASKS_PATH #Need to rerun mask generator for create mask and relative path

LoadTestImagesForCrop = TEST_OR_PATH
LoadTestLabelsForCrop = TEST_MASKS_PATH

SaveCropImages =  str(TRAIN_VALID_CROP_OR_PATH) #Create this folder in config (unique folder)
SaveCropMasks = str(TRAIN_VALID_CROP_MASKS_PATH) #Create this folder in config (unique folder)

SaveCropImagesRed =  str(ALL_IMAGES) #Create this folder in config (unique folder)
SaveCropMasksRed = str(ALL_MASKS) #Create this folder in config (unique folder)

SaveCropImages =  str(TRAIN_VALID_CROP_OR_PATH) #Create this folder in config (unique folder)
SaveCropMasks = str(TRAIN_VALID_CROP_MASKS_PATH) #Create this folder in config (unique folder)

SaveCropTestImages =  str(TEST_CROP_OR_PATH) #Create this folder in config (unique folder)
SaveCropTestMasks = str(TEST_CROP_MASKS_PATH) #Create this folder in config (unique folder)

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


def make_cropper(train_valid = True):

    if train_valid:
        ix = 0
        image_ids = check_output(["ls", LoadImagesForCrop]).decode("utf8").split()
        number = [int(num.split('.')[0]) for num in image_ids]
        number.sort()
        for ax_index, num in tqdm(enumerate(number),total=len(number)):
            image_id = str(num)+'.'+'TIF'
            image, mask = reader.loadImage(image_id, 'train', 'original', True)
            CroppedImages, CroppedMasks = cropper(image, mask)
            
            for i in range(0,NumCropped):
            
                crop_imgs_dir = SaveCropImages +'/'+ '{}.TIF'.format(ix)
                crop_masks_dir = SaveCropMasks +'/'+ '{}.TIF'.format(ix) 

                plt.imsave(fname= crop_imgs_dir, arr = CroppedImages[i])
                plt.imsave(fname= crop_masks_dir,arr = CroppedMasks[i])
                
                ix +=1
                
    else:
        ix = 0
        image_ids = check_output(["ls", LoadTestImagesForCrop]).decode("utf8").split()
        number = [int(num.split('.')[0]) for num in image_ids]
        number.sort()
        for ax_index, num in tqdm(enumerate(number),total=len(number)):
            image_id = str(num)+'.'+'TIF'
            image, mask = reader.loadImage(image_id, 'test', 'original', True)
            CroppedImages, CroppedMasks = cropper(image, mask)
            
            for i in range(0,NumCropped):
            
                crop_imgs_dir = SaveCropTestImages +'/'+ '{}.TIF'.format(ix)
                crop_masks_dir = SaveCropTestMasks +'/'+ '{}.TIF'.format(ix) 

                plt.imsave(fname= crop_imgs_dir, arr = CroppedImages[i])
                plt.imsave(fname= crop_masks_dir,arr = CroppedMasks[i])
                
                ix +=1    
    return
    
    
    
    
def cropper_red(image, mask):
    
    if image.shape[0] == 1200:
        
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


        CroppedImgs = np.zeros((NumCropped, YCropSize, XCropSize, 3), np.uint8)
        CroppedMasks = np.zeros((NumCropped, YCropSize, XCropSize, 3), np.uint8)
        idx = 0

        for i in range(0, 4):
            for j in range(0, 3):

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

        return CroppedImgs, CroppedMasks, NumCropped
    
    else:
        
        IMG_WIDTH = 2272
        IMG_HEIGTH = 1704

        XCropSize = 512
        YCropSize = 512

        XCropCoord = 284
        YCropCoord = 284

        XCropNum = int(IMG_WIDTH/XCropCoord)
        YCropNum = int(IMG_HEIGTH/YCropCoord)

        NumCropped = int(IMG_WIDTH/XCropCoord * IMG_HEIGTH/YCropCoord)

        YShift = YCropSize - YCropCoord
        XShift = XCropSize - XCropCoord

        x_coord = [XCropCoord*i for i in range(0, XCropNum+1)]
        y_coord = [YCropCoord*i for i in range(0, YCropNum+1)]

        CroppedImgs = np.zeros((NumCropped, YCropSize, XCropSize, 3), np.uint8)
        CroppedMasks = np.zeros((NumCropped, YCropSize, XCropSize, 3), np.uint8)
        idx = 0

        for i in range(0, 8):
            for j in range(0, 6):

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

        return CroppedImgs, CroppedMasks, NumCropped

def make_cropper_red(train_valid = True):

    ix = 0
    image_ids = check_output(["ls", LoadImagesForCrop]).decode("utf8").split()
    number = [int(num.split('.')[0]) for num in image_ids]
    number.sort()
    for ax_index, num in tqdm(enumerate(number),total=len(number)):
        image_id = str(num)+'.'+'TIF'
        image, mask = reader.loadImage(image_id, 'train', 'original', True)
        CroppedImages, CroppedMasks = cropper_red(image, mask)
        
        for i in range(0,NumCropped):
            #Save in all images and all masks
            crop_imgs_dir = SaveCropImagesRed +'/'+ '{}.TIF'.format(ix)
            crop_masks_dir = SaveCropMasksRed +'/'+ '{}.TIF'.format(ix) 

            plt.imsave(fname= crop_imgs_dir, arr = CroppedImages[i])
            plt.imsave(fname= crop_masks_dir,arr = CroppedMasks[i])
            
            ix +=1
    
    

make_cropper_red(train_valid = True)


