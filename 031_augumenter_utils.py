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

import cv2
import glob
import sys
import numpy as np
import imageio
import random
from skimage import transform
import matplotlib.pyplot as plt
from tqdm import tqdm
from subprocess import check_output
from importlib import import_module
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.morphology import watershed, remove_small_holes, remove_small_objects, label, erosion
from skimage.feature import peak_local_max
from skimage.transform import rotate
from albumentations import (RandomCrop,CenterCrop,ElasticTransform,RGBShift,Rotate,
    Compose, ToFloat, FromFloat, RandomRotate90, Flip, OneOf, MotionBlur, MedianBlur, Blur,Transpose,
    ShiftScaleRotate, OpticalDistortion, GridDistortion, RandomBrightnessContrast, VerticalFlip, HorizontalFlip,
    
    HueSaturationValue,
)

reader = import_module('020_loader')

from config_script import *    

def data_aug(image,mask,angel=30):
    flip =  random.random()
    Gauss_noise = random.randint(0, 1)
    lower = random.randint(0, 1)
    translate_down = random.randint(0, 1)
    S_P = random.randint(0, 1)
    resize = 1 if random.random() > 0.7 else 0

    rows,cols,ch= image.shape

    if resize:
        res = np.random.uniform(low=0.5, high=0.9)
        scaled_img = cv2.resize(image,(int(rows*res),int(cols*res))) # scale image if you want resize the input andoutput image must be the same
        scaled_lbl = cv2.resize(mask,(int(rows*res),int(cols*res)))
        sh, sw = scaled_img.shape[:2] # get h, w of scaled image
        center_y = int(rows/2 - sh/2)
        center_x = int(cols/2 - sw/2)
        image = np.zeros(image.shape, dtype=np.uint8) # using img.shape to obtain #channels
        mask = np.zeros(image.shape, dtype=np.uint8)
        image[center_y:center_y+sh, center_x:center_x+sw] = scaled_img
        mask[center_y:center_y+sh, center_x:center_x+sw] = scaled_lbl

        afine_tf = transform.AffineTransform(shear=None,rotation=None)
        image = transform.warp(image, inverse_map=afine_tf,mode='constant')
        mask = transform.warp(mask, inverse_map=afine_tf,mode='constant')        

    else:
        sh = random.random()/2-0.25
        rotate_angel = random.random()/180*np.pi*angel
        afine_tf = transform.AffineTransform(shear=sh,rotation=rotate_angel)
        image = transform.warp(image, inverse_map=afine_tf,mode='constant')
        mask = transform.warp(mask, inverse_map=afine_tf,mode='constant')

    if (S_P) & (not Gauss_noise):
        s_vs_p = 0.02
        amount = 0.004
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape[0:2]]
        image[coords] = [255,255,255]

    if Gauss_noise:
        mean = image.mean()
        std = image.std()*0.5
        noise = np.random.normal(mean, std, image.shape)
        noise_img = np.zeros((rows, cols, ch), dtype=np.uint8)
        noise_img = noise
        noise.astype(int)
        image = image + noise
        image = np.clip(image, 0, 255)

    if lower:
        image *= 0.4

    if translate_down:
        to_right =random.randint(-40,40)
        to_up = random.randint(0,150)
        M = np.float32([[1,0,to_right],[0,1,to_up]])
        dst = cv2.warpAffine(image,M,(cols,rows))
        dst_mask = cv2.warpAffine(mask,M,(cols,rows))
        image = dst[:,::-1]
        mask = dst_mask[:,::-1]

    if flip <= 0.25:
        image = image

    elif 0.25 < flip <= 0.50:
        image = cv2.flip( image, 0 )
        mask = cv2.flip( mask, 0 )
    elif 0.50 < flip <= 0.75:
        image = cv2.flip( image, 1 )
        mask = cv2.flip( mask, 1 )
    elif flip > 0.75:
        image = cv2.flip( image, -1 )
        mask = cv2.flip( mask, -1 )

    return image, mask

def data_aug_crop(image, mask, image_id):

    resizer = random.random()
    flip =  random.random()
    Gauss_noise = random.random()
    rotation = random.randint(0, 1)
    S_P = random.randint(0, 1)
    lower = random.randint(0, 1)

    rows,cols,ch = image.shape
    rowsm,colsm,chm = mask.shape
    image_id = int(image_id.split('.')[0])

    if (resizer <= 0.15) & (image_id < 2580):

        res = 0.5  
        scaled_image = cv2.resize(image,(int(cols*res),int(rows*res))) # scale image if you want resize the input andoutput image must be the same
        scaled_mask = cv2.resize(mask,(int(cols*res),int(rows*res)))
        bordersize = rows//4
        b, g, r = cv2.split(image)
        blu = b.mean()
        green = g.mean()
        red = r.mean()
        image=cv2.copyMakeBorder(scaled_image, top=bordersize, bottom=bordersize, left=bordersize, 
                                 right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[blu,green,red])
        mask=cv2.copyMakeBorder(scaled_mask, top=bordersize, bottom=bordersize, left=bordersize, 
                                right=bordersize, borderType= cv2.BORDER_CONSTANT)

    if rotation:

        augmentation = rotating(p = 1)
        data = {"image": image, "mask": mask}
        augmented = augmentation(**data)
        image, mask = augmented["image"], augmented["mask"]

    if (S_P) & (not Gauss_noise):

        s_vs_p = 0.02
        amount = 0.004
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape[0:2]]
        image[coords] = [255,255,255]


    if Gauss_noise < 0.20:
        mean = image.mean()
        std = image.std()*0.3
        noise = np.random.normal(mean, std, image.shape)
        noise_img = np.zeros((rows, cols, ch), dtype=np.uint8)
        noise_img = noise.astype(int)
        image = image + noise_img
        image = np.clip(image, 0, 255)
        image = image.astype(int)


    if flip <= 0.25:
        image = image

    elif 0.25 < flip <= 0.50:
        image = cv2.flip( image, 0 )
        mask = cv2.flip( mask, 0 )
    elif 0.50 < flip <= 0.75:
        image = cv2.flip( image, 1 )
        mask = cv2.flip( mask, 1 )
    elif flip > 0.75:
        image = cv2.flip( image, -1 )
        mask = cv2.flip( mask, -1 )

        if lower:
            np.multiply(image, 0.6, out = image, casting = 'unsafe')

    return image, mask


def elastic_aug(image,mask, image_id):
    image_id = int(image_id.split('.')[0])
    S_P = random.random()
    resize = random.random()
    generic_transf = random.random()
    elastic = random.random()

    rows,cols,ch = image.shape
    rowsm,colsm,chm = mask.shape


    if generic_transf < 0.65:

        b, g, r = cv2.split(image)
        blu = b.mean()
        green = g.mean()
        red = r.mean()
        color = green*0.05

        augmentation = strong_tiff_aug(color, p = 1)
        data = {"image": image, "mask": mask}
        augmented = augmentation(**data)
        image, mask = augmented["image"], augmented["mask"]

        if (S_P) >= 0.9:

            s_vs_p = 0.02
            amount = 0.004
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape[0:2]]
            image[coords] = [255,255,255]

        return image, mask



    if elastic > 0.6:

        alfa = random.choice([50,50,50, 70, 70, 70])
        alfa_affine = random.choice([50,50,50, 75, 75])
        sigma = random.choice([30, 30, 30, 30, 40, 40, 20])
        elastic = elastic_def_beta(alfa, alfa_affine, sigma, p=1)
        data = {"image": image, "mask": mask}
        augmented = elastic(**data)
        image, mask = augmented["image"], augmented["mask"]

        if (S_P) >= 0.9:

            s_vs_p = 0.02
            amount = 0.004
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape[0:2]]
            image[coords] = [255,255,255]


        return image, mask


    if (resize <= 0.65) & (image_id < 2580):

        res = 0.5  
        scaled_image = cv2.resize(image,(int(cols*res),int(rows*res))) # scale image if you want resize the input andoutput image must be the same
        scaled_mask = cv2.resize(mask,(int(cols*res),int(rows*res)))
        bordersize = rows//4
        b, g, r = cv2.split(image)
        blu = b.mean()
        green = g.mean()
        red = r.mean()
        image=cv2.copyMakeBorder(scaled_image, top=bordersize, bottom=bordersize, left=bordersize, 
                                 right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[blu,green,red] )
        mask=cv2.copyMakeBorder(scaled_mask, top=bordersize, bottom=bordersize, left=bordersize, 
                                right=bordersize, borderType= cv2.BORDER_CONSTANT)
    else:

        augmentation = rotating(p = 1)
        data = {"image": image, "mask": mask}
        augmented = augmentation(**data)
        image, mask = augmented["image"], augmented["mask"]


    return image, mask


def make_data_augmentation(split_num, SaveAugImages, SaveAugMasks, OnCropped = True, Elastic = False):

    if (OnCropped==True) & (Elastic==True):

        SaveAugImages = str(TRAIN_VALID_CROP_OR_PATH) + '/'
        SaveAugMasks = str(TRAIN_VALID_CROP_MASKS_PATH) + '/'
        image_ids = check_output(["ls", TRAIN_VALID_CROP_OR_PATH]).decode("utf8").split()
        ix = len(image_ids)

        for ax_index, image_id in tqdm(enumerate(image_ids),total=len(image_ids)):

            ID = int(image_id.split('.')[0])
            image, mask = reader.loadImage(image_id, 'train', 'crop', True)
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([img_gray],[0],None,[256],[0,256])
            mask_to_count = np.squeeze(mask[:,:,0:1])
            mask_image, n_objs_mask = ndimage.label(mask_to_count, np.ones((3,3)))

            if (n_objs_mask == 0) & (hist[0] < 3000):
                split_num = 1
            elif (0 <= n_objs_mask <= 10):
                split_num = 3
            else:
                split_num = 4
            if (ID > 2580) | (hist[0] > 3000):            
                split_num = split_num + 2

            for i in range(split_num):

                new_image, new_mask = elastic_aug(image,mask,image_id)
                new_image = new_image*1./255
                new_mask = new_mask*1./255
                new_image.astype(float)
                new_mask.astype(float)
                aug_img_dir = SaveAugImages + '{}.TIF'.format(ix)
                aug_mask_dir = SaveAugMasks + '{}.TIF'.format(ix)
                ix +=1
                plt.imsave(fname=aug_img_dir, arr = new_image)
                plt.imsave(fname=aug_mask_dir,arr = new_mask)

    elif (OnCropped==True) & (Elastic==False):

        SaveAugImages = str(TRAIN_VALID_CROP_OR_PATH) + '/'
        SaveAugMasks = str(TRAIN_VALID_CROP_MASKS_PATH) + '/'
        image_ids = check_output(["ls", TRAIN_VALID_CROP_OR_PATH]).decode("utf8").split()
        ix = len(image_ids)

        for ax_index, image_id in tqdm(enumerate(image_ids),total=len(image_ids)):

            ID = int(image_id.split('.')[0])
            image, mask = reader.loadImage(image_id, 'train', 'crop', True)

            for i in range(split_num):
                new_image, new_mask = data_aug_crop(image,mask,image_id)
                new_image = new_image*1./255
                new_mask = new_mask*1./255
                new_image.astype(float)
                new_mask.astype(float)                  
                aug_img_dir = SaveAugImages + '{}.TIF'.format(ix)
                aug_mask_dir = SaveAugMasks + '{}.TIF'.format(ix)
                ix +=1              
                plt.imsave(fname=aug_img_dir, arr = new_image)
                plt.imsave(fname=aug_mask_dir,arr = new_mask)                

    else:
        image_ids = check_output(["ls", TRAIN_VALID_OR_PATH]).decode("utf8").split()
        ix = len(image_ids)

        for ax_index, image_id in tqdm(enumerate(image_ids),total=len(image_ids)):
            image, mask = reader.loadImage(image_id, 'train', 'original', True)

            for i in range(split_num):
                new_image, new_mask = data_aug(image,mask,angel=30)
                aug_img_dir = TRAIN_VALID_AUG_OR_PATH / '{}.TIF'.format(ix)
                aug_mask_dir = TRAIN_VALID_AUG_MASKS_PATH / '{}.TIF'.format(ix)
                ix +=1 

                plt.imsave(fname=str(aug_img_dir), arr = new_image)
                plt.imsave(fname=str(aug_mask_dir),arr = new_mask)
#################################################################################################
#################################################################################################                
#############################################RED IMAGES AUGUMENTATIONS###########################
#################################################################################################
#################################################################################################
def shifter(p=.5):

    return Compose([
        ToFloat(),

        #ROTATION
        Rotate(limit=180, interpolation=1, border_mode=cv2.BORDER_CONSTANT,
               always_apply=False, p=0.75),
#         #FLIP
        OneOf([
            VerticalFlip(p = 0.6),
            HorizontalFlip(p = 0.6),
                ], p=p),
        
        FromFloat(dtype='uint8', max_value=255.0),
        
        ], p=p)
        
def lookup_tiff_aug(p = 0.5):

    return Compose([

        ToFloat(),
        
        #LOOKUP TABLE    
        OneOf([ 
        # HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=(-0.05,0.05), p=0.7),
        RandomBrightnessContrast(brightness_limit=0,contrast_limit=(-0.5,+0.1), p=1),
        
            ], p=p),
        
        FromFloat(dtype='uint8', max_value=255.0),

            ], p=p)

def Gaussian(p=.5, blur_limit = 11):
    return Compose([
        ToFloat(),

            OneOf([
            Blur(blur_limit=blur_limit, p=1),
        ], p=1),
        
        FromFloat(dtype='uint8', max_value=255.0),
        

    ], p=p)
    
def distortion(p = 0.5):

    return Compose([

        ToFloat(),
        
        #LOOKUP TABLE    
        OneOf([ 

            GridDistortion(num_steps=4, distort_limit=0.5, interpolation=1, border_mode=cv2.BORDER_CONSTANT, p=1)
        
            ], p=p),
        
        FromFloat(dtype='uint8', max_value=255.0),

            ], p=p)
    
def elastic_def(alpha, alpha_affine, sigma, p=.5):

    return Compose([
        ToFloat(),
        

        ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=alpha_affine, interpolation=1, 
                         border_mode=cv2.BORDER_CONSTANT, always_apply=False,
                         approximate=False, p=1),
        
        FromFloat(dtype='uint8', max_value=255.0),
        

    ], p=p)
                
def red_aug(image ,mask, image_id):
    
    image_id = int(image_id.split('.')[0])
    gaussian = random.random()
    generic_transf = random.random()
    elastic = random.random()
    resize = random.random()
    distorted = random.random()
    
    minimum = mask[:,:,1:2].min()
    maximum = mask[:,:,1:2].max()
    rows,cols,ch = image.shape
    rowsm,colsm,chm = mask.shape

    if generic_transf < 0.60:
       
        augmentation = lookup_tiff_aug(p = 0.8)
        data = {"image": image}
        augmented = augmentation(**data)
        image = augmented["image"]
        
        augmentation = shifter(p = 0.7)
        data = {"image": image, "mask": mask}
        augmented = augmentation(**data)
        image, mask = augmented["image"], augmented["mask"] 
        
        mask[:,:,1:2] =np.clip(mask[:,:,1:2], minimum, maximum)
        
        if gaussian <= 0.30:
            
            gaussian_blur = Gaussian(p=1, blur_limit =15)
            data = {"image": image}
            augmented = gaussian_blur(**data)
            image = augmented["image"] 
            
        return image, mask
    
    if elastic < 0.95:
        
        alfa = random.choice([50,50, 60, 60, 65, 65, 65, 70])
        alfa_affine = random.choice([35, 40,40, 45, 50])
        sigma = random.choice([15,20, 20, 25, 25, 25, 30 ,30])
        elastic = elastic_def(alfa, alfa_affine, sigma, p=1)
        data = {"image": image, "mask": mask}
        augmented = elastic(**data)
        image, mask = augmented["image"], augmented["mask"] 
        
        mask[:,:,1:2] =np.clip(mask[:,:,1:2], minimum, maximum)
        
        if gaussian <= 0.25:
            
            gaussian_blur = Gaussian(p=1, blur_limit =11)
            data = {"image": image}
            augmented = gaussian_blur(**data)
            image = augmented["image"] 


        return image, mask
        
    if distorted  < 1:
        
        augmentation = shifter(p = 0.95)
        data = {"image": image, "mask": mask}
        augmented = augmentation(**data)
        image, mask = augmented["image"], augmented["mask"] 
        
        mask[:,:,1:2] =np.clip(mask[:,:,1:2], minimum, maximum)
        
        if gaussian <= 0.15:
            
            gaussian_blur = Gaussian(p=1, blur_limit =11)
            data = {"image": image}
            augmented = gaussian_blur(**data)
            image = augmented["image"] 
             
        return image, mask

                
                
def make_augumentation_on_red(split_num, SaveAugImages, SaveAugMasks, start = 'begin'):

    SaveAugImages = str(ALL_IMAGES) + '/'
    SaveAugMasks = str(ALL_MASKS) + '/'
    
    image_ids_name = check_output(["ls", ALL_IMAGES]).decode("utf8").split()
    ix = [x for x in range(2964)]
    #1632 fo all ( big)
    ix.sort()
    image_ids = [str(x)+'.TIF' for x in ix]
    
    if start == 'begin':
    
         ix = len(image_ids)
         
    elif start == 'continue':
    
         ix = len(image_ids_name)

    for ax_index, image_id in tqdm(enumerate(image_ids),total=len(image_ids)):
    
        ID = int(image_id.split('.')[0])
    
        image, mask = reader.loadImage(image_id, 'train', 'all_images', True)
        
        labels_tar, nlabels_tar = ndimage.label(np.squeeze(mask[:,:,0:1]))
        
        if ix > 18000:
            print('limit of memory')
            break
        
                
        # if (ID>432):

            # for i in range(split_num):

                 # new_image, new_mask = red_aug(image,mask,image_id)
            
                 # aug_img_dir = SaveAugImages + '{}.TIF'.format(ix)
                 # aug_mask_dir = SaveAugMasks + '{}.TIF'.format(ix)
                 # ix +=1
                 # plt.imsave(fname=aug_img_dir, arr = new_image)
                 # plt.imsave(fname=aug_mask_dir,arr = new_mask)
                 
        # else:
        
        for i in range(split_num):

             new_image, new_mask = red_aug(image,mask,image_id)
        
             aug_img_dir = SaveAugImages + '{}.TIF'.format(ix)
             aug_mask_dir = SaveAugMasks + '{}.TIF'.format(ix)
             ix +=1
             plt.imsave(fname=aug_img_dir, arr = new_image)
             plt.imsave(fname=aug_mask_dir,arr = new_mask)
#################################################################################################
#################################################################################################                
#############################################RED IMAGES AUGUMENTATIONS###########################
#################################################################################################
#################################################################################################
  


def strong_tiff_aug(color, p = 0.5):
    return Compose([

        ToFloat(),

        #ROTATION
            Rotate(limit=90, interpolation=1, border_mode=4, always_apply=False, p=0.75),
#         #FLIP
OneOf([
    VerticalFlip(p = 0.6),
    HorizontalFlip(p = 0.6)
    ], p=0.6),

#         #NOISE
OneOf([
    Blur(blur_limit=10, p=1),
    ], p=0.25),


#LOOKUP TABLE    
OneOf([ 
    HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.15, val_shift_limit=0.1, p=0.33),
    RandomBrightnessContrast(brightness_limit=(-color, 0), p=0.5),
    ], p=0.6),

FromFloat(dtype='uint8', max_value=255.0),


], p=p)


def elastic_def_beta(alpha, alpha_affine, sigma, p=.5):
    return Compose([
        ToFloat(),

        ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=alpha_affine, interpolation=1, border_mode=4, 
                             always_apply=False, approximate=False, 
                             p=1),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=(0, 0),
                             interpolation=1, border_mode=4, always_apply=False, p=0.3),

            FromFloat(dtype='uint8', max_value=255.0),


        ], p=p)


def rotating(p=.5):
    return Compose([
        ToFloat(),

        ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=45,
                             interpolation=1, border_mode=4, always_apply=False, p=1),

            FromFloat(dtype='uint8', max_value=255.0),

            ], p=p)
            
            
        
#################################################################################################
#################################################################################################                
#############################################YELLOW IMAGES AUGUMENTATIONS###########################
#################################################################################################
#################################################################################################



def lookup_tiff_aug_y(p = 0.5):
    return Compose([

        ToFloat(),
        
        #LOOKUP TABLE    
        OneOf([ 
#         HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=[0.25,0.26], p=1),
        RandomBrightnessContrast(brightness_limit=0,contrast_limit=(-0.7,0.0), p=0.7),
        HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=0.05, p=0.7),
        # RGBShift(r_shift_limit=0, g_shift_limit=0.15, b_shift_limit=0, p=0.15),
        # RGBShift(r_shift_limit=0.15, g_shift_limit=0, b_shift_limit=0, p=0.15),
        # RGBShift(r_shift_limit=0.15, g_shift_limit=0.15, b_shift_limit=0, p=0.15),

            ], p=p),
        
        FromFloat(dtype='uint8', max_value=255.0),

            ], p=p)

def shifter_RGB_y(p = 0.5):
    return Compose([

        ToFloat(),
        
        #LOOKUP TABLE    
        OneOf([ 
        RGBShift(r_shift_limit=[0.05,0.06], g_shift_limit=[0.04,0.045], b_shift_limit=0, p=1),
            ], p=p),
        
        FromFloat(dtype='uint8', max_value=255.0),

            ], p=p)

def shifter_y(p=.5):
    return Compose([
        ToFloat(),

        #ROTATION
        Rotate(limit=180, interpolation=1, border_mode=4, always_apply=False, p=0.75),
#         #FLIP
        OneOf([
            VerticalFlip(p = 0.6),
            HorizontalFlip(p = 0.6),
                ], p=p),
        
        FromFloat(dtype='uint8', max_value=255.0),
        
        ], p=p)


def elastic_def_y(alpha, alpha_affine, sigma, p=.5):
    return Compose([
        ToFloat(),

        ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=alpha_affine, interpolation=1, border_mode=4, 
                             always_apply=False, approximate=False, 
                             p=1),
        ShiftScaleRotate(shift_limit=0.10, scale_limit=0, rotate_limit=(0, 0),
                         interpolation=1, border_mode=4, always_apply=False, p=0.3),
        
        FromFloat(dtype='uint8', max_value=255.0),
        

    ], p=p)

def edges_aug_y(p = 0.5):
    return Compose([

        ToFloat(),
        
        #LOOKUP TABLE    
        OneOf([ 
        HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit=0.10, val_shift_limit=0.1, p=0.75),
        RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.4,p=0.75),
        ], p=0.6),
        
        FromFloat(dtype='uint8', max_value=255.0),

    ], p=p)


def Gaussian_y(p=.5, blur_limit = 25):
    return Compose([
        ToFloat(),

            OneOf([
            Blur(blur_limit=25, p=1),
        ], p=1),
        
        FromFloat(dtype='uint8', max_value=255.0),
        

    ], p=p)




def yellow_data_aug(image ,mask, nlabels_tar, minimum, maximum):
    
    gaussian = random.random()
    generic_transf = random.random()
    elastic = random.random()
    resize = random.random()
    RGB = random.random()
    
    rows,cols,ch = image.shape
    rowsm,colsm,chm = mask.shape
    
    # if lower < 0.08:
        # image = image.astype(np.float)
        # image *= 0.4
        # image = image.astype(np.uint8)
        
        # return image, mask

    if (RGB < 0.05) & (nlabels_tar > 2):
       
        augmentation = shifter_RGB_y(p = 1)
        data = {"image": image}
        augmented = augmentation(**data)
        image = augmented["image"]
        
        augmentation = shifter_y(p = 0.5)
        data = {"image": image, "mask": mask}
        augmented = augmentation(**data)
        image, mask = augmented["image"], augmented["mask"] 
        
        mask[:,:,1:2] =np.clip(mask[:,:,1:2], minimum, maximum)
        
        # if gaussian <= 0.10:
            
            # gaussian_blur = Gaussian_y(p=1, blur_limit = 15)
            # data = {"image": image}
            # augmented = gaussian_blur(**data)
            # image = augmented["image"] 
            

        return image, mask
    
    
    if resize <= 0.5:

        res = 0.5  
        scaled_image = cv2.resize(image,(int(cols*res),int(rows*res))) # scale image if you want resize the input andoutput image must be the same
        scaled_mask = cv2.resize(mask,(int(cols*res),int(rows*res)))
        bordersize = rows//4
        b, g, r = cv2.split(image)
        blu = b.mean()
        green = g.mean()
        red = r.mean()
        image=cv2.copyMakeBorder(scaled_image, top=bordersize, bottom=bordersize, left=bordersize, 
                             right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[blu,green,red])
        mask=cv2.copyMakeBorder(scaled_mask, top=bordersize, bottom=bordersize, left=bordersize, 
                            right=bordersize, borderType= cv2.BORDER_CONSTANT)
                            
        mask[:,:,1:2] =np.clip(mask[:,:,1:2], minimum, maximum)
        
        return image, mask
        
        
    #65 before
    if generic_transf < 0.65:
       
        augmentation = lookup_tiff_aug_y(p = 0.7)
        data = {"image": image}
        augmented = augmentation(**data)
        image = augmented["image"]
        
        augmentation = shifter_y(p = 0.7)
        data = {"image": image, "mask": mask}
        augmented = augmentation(**data)
        image, mask = augmented["image"], augmented["mask"] 
        
        mask[:,:,1:2] =np.clip(mask[:,:,1:2], minimum, maximum)
        
        if gaussian <= 0.33:
            
            gaussian_blur = Gaussian_y(p=1, blur_limit = 15)
            data = {"image": image}
            augmented = gaussian_blur(**data)
            image = augmented["image"] 
            

        return image, mask
    

    if elastic < 0.9:
        
        alfa = random.choice([30, 30, 40, 40, 40 , 50, 60])
        alfa_affine = random.choice([40, 50, 50, 75, 75])
        sigma = random.choice([20, 30, 30, 40, 50])
        elastic = elastic_def_y(alfa, alfa_affine, sigma, p=1)
        data = {"image": image, "mask": mask}
        augmented = elastic(**data)
        image, mask = augmented["image"], augmented["mask"]      

        mask[:,:,1:2] =np.clip(mask[:,:,1:2], minimum, maximum)        

        return image, mask

    else:
        
        augmentation = shifter_y(p = 1)
        data = {"image": image, "mask": mask}
        augmented = augmentation(**data)
        image, mask = augmented["image"], augmented["mask"]
        
        mask[:,:,1:2] =np.clip(mask[:,:,1:2], minimum, maximum)
        
        return image, mask
        
        
        
        
def make_augumentation_on_yellow(split_num, SaveAugImages, SaveAugMasks, start = 'begin'):


    id_edges = [336, 765, 1209, 2313, 2314,2336]

    SaveAugImages = str(ALL_IMAGES) + '/'
    SaveAugMasks = str(ALL_MASKS) + '/'
    
    image_ids_name = check_output(["ls", ALL_IMAGES]).decode("utf8").split()
    ix = [x for x in range(2556)]
    #2664 before
    ix.sort()
    image_ids = [str(x)+'.TIF' for x in ix]
    
    if start == 'begin':
    
         ix = len(image_ids)
         
    elif start == 'continue':
    
         ix = len(image_ids_name)

    
    for ax_index, image_id in tqdm(enumerate(image_ids),total=len(image_ids)):
    
        if ax_index > 2304:
           
           split_num =14

        if ix > 18550:
            print('limit of memory')
            break
              
        ID = int(image_id.split('.')[0])
    
        image, mask = reader.loadImage(image_id, 'train', 'all_images', True)
        minimum = mask[:,:,1:2].min()
        maximum = mask[:,:,1:2].max()
        labels_tar, nlabels_tar = ndimage.label(np.squeeze(mask[:,:,0:1]))

        # target = mask[:,:,0:1].astype(bool)
        # target = remove_small_objects(target,min_size = 30)    
        # mask[:,:,0:1] = target.astype(np.uint8)*255
        
        if ID in id_edges:
            
            print(ID, ix)
                     #80 before
            for i in range(80):
                
                image, mask = reader.loadImage(image_id, 'train', 'all_images', True)

                augmentation = edges_aug_y(p = 1)
                data = {"image": image}
                augmented = augmentation(**data)
                new_image = augmented["image"]
                
                augmentation = shifter_y(p = 0.8)
                data = {"image": new_image, "mask": mask}
                augmented = augmentation(**data)
                new_image, new_mask = augmented["image"], augmented["mask"]
                
                new_mask[:,:,1:2] =np.clip(new_mask[:,:,1:2], minimum, maximum)
        
                aug_img_dir = SaveAugImages + '{}.TIF'.format(ix)
                aug_mask_dir = SaveAugMasks + '{}.TIF'.format(ix)
                ix +=1
                plt.imsave(fname=aug_img_dir, arr = new_image)
                plt.imsave(fname=aug_mask_dir,arr = new_mask)
                #35 before
            for i in range(35):
                
                image, mask = reader.loadImage(image_id, 'train', 'all_images', True)
                 
                alfa = random.choice([30,30,30, 40])
                alfa_affine = random.choice([20,20,20,30, 40, 40])
                sigma = random.choice([20, 20, 20, 20, 30, 30, 15])
                
                elastic = elastic_def_y(alfa, alfa_affine, sigma, p=1)
                data = {"image": image, "mask": mask}
                augmented = elastic(**data)
                new_image, new_mask = augmented["image"], augmented["mask"] 
                
                new_mask[:,:,1:2] =np.clip(new_mask[:,:,1:2], minimum, maximum)

                aug_img_dir = SaveAugImages + '{}.TIF'.format(ix)
                aug_mask_dir = SaveAugMasks + '{}.TIF'.format(ix)
                ix +=1
                plt.imsave(fname=aug_img_dir, arr = new_image)
                plt.imsave(fname=aug_mask_dir,arr = new_mask)
                #39 before
            for blur in range(1 , 39, 3):
                
                image, mask = reader.loadImage(image_id, 'train', 'all_images', True)
                
                blur_limit = blur
                gaussian_blur = Gaussian_y(p = 1, blur_limit= blur_limit)
                data = {"image": image}
                augmented = gaussian_blur(**data)
                new_image = augmented["image"]   

                aug_img_dir = SaveAugImages + '{}.TIF'.format(ix)
                aug_mask_dir = SaveAugMasks + '{}.TIF'.format(ix)
                ix +=1
                plt.imsave(fname=aug_img_dir, arr = new_image)
                plt.imsave(fname=aug_mask_dir,arr = mask)
                
                
                

        else:
    
            for i in range(split_num):

                new_image, new_mask = yellow_data_aug(image, mask, nlabels_tar, minimum, maximum)
                    
                aug_img_dir = SaveAugImages + '{}.TIF'.format(ix)
                aug_mask_dir = SaveAugMasks + '{}.TIF'.format(ix)
                ix +=1
                plt.imsave(fname=aug_img_dir, arr = new_image)
                plt.imsave(fname=aug_mask_dir,arr = new_mask)






