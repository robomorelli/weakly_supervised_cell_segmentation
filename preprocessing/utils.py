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

from albumentations import (RandomCrop,CenterCrop,ElasticTransform,RGBShift,Rotate,
    Compose, ToFloat, FromFloat, RandomRotate90, Flip, OneOf, MotionBlur, MedianBlur, Blur,Transpose,
    ShiftScaleRotate, OpticalDistortion, GridDistortion, RandomBrightnessContrast, VerticalFlip, HorizontalFlip,
    
    HueSaturationValue,
)

from config import *

def read_masks(path, image_id):
    mask = cv2.imread(path + image_id)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    return mask

def read_images(path, image_id):
    img = cv2.imread(path + image_id)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def read_image_masks(image_id, images_path, masks_path):
    x = cv2.imread(images_path + image_id)
    image = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(masks_path + image_id)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    return image, mask

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

def data_aug_crop(image, mask, name):

    resize_flag = True if '100x' in name else False
    resizer = random.random()
    flip =  random.random()
    Gauss_noise = random.random()
    rotation = random.randint(0, 1)
    S_P = random.randint(0, 1)
    lower = random.randint(0, 1)

    rows,cols,ch = image.shape

    if (resizer <= 0.15) & (resize_flag):

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


def elastic_aug(image,mask, name):
    resize_flag = True if '100x' in name else False
    S_P = random.random()
    resize = random.random()
    generic_transf = random.random()
    elastic = random.random()

    rows,cols,ch = image.shape

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

    if (resize <= 0.65) & (resize_flag):

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


def make_data_augmentation(split_num, images_path, masks_path,
                           saved_images_path, saved_masks_path,
                           OnCropped = True, Elastic = False):

    Images = str(images_path).replace('preprocessing/', '') + '/'
    Masks = str(masks_path).replace('preprocessing/', '') + '/'
    SaveAugImages = str(saved_images_path).replace('preprocessing/', '') + '/'
    SaveAugMasks = str(saved_masks_path).replace('preprocessing/', '') + '/'
    images_name = check_output(["ls", Images]).decode("utf8").split()

    for ax_index, name_ext in tqdm(enumerate(images_name),total=len(images_name)):
        name = name_ext.split('.')[0]
        ext = name_ext.split('.')[1]

        image, mask = read_image_masks(name_ext, Images, Masks)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([img_gray],[0],None,[256],[0,256])
        mask_to_count = np.squeeze(mask[:,:,0:1])
        mask_image, n_objs_mask = ndimage.label(mask_to_count, np.ones((3,3)))

        if (n_objs_mask == 0) & (hist[0] < 3000):
            split_num = 1
        elif (0 < n_objs_mask <= 10):
            split_num = max(split_num-2, 2)
        else:
            split_num = split_num

        for i in range(split_num):

            if (OnCropped == True) & (Elastic == True):
                new_image, new_mask = elastic_aug(image,mask,name)
            elif (OnCropped == True) & (Elastic == False):
                new_image, new_mask = data_aug_crop(image, mask, name)
            else:
                new_image, new_mask = data_aug(image, mask, angel=30)

            new_image = new_image*1./255
            new_mask = new_mask*1./255
            new_image.astype(float)
            new_mask.astype(float)

            aug_img_dir = SaveAugImages + '{}_{}.{}'.format(name, i, ext)
            aug_mask_dir = SaveAugMasks + '{}_{}.{}'.format(name, i, ext)
            plt.imsave(fname=aug_img_dir, arr = new_image)
            plt.imsave(fname=aug_mask_dir,arr = new_mask)
######################################################################################
######################################################################################
#############################################TRANSOFRMATION###########################
######################################################################################
######################################################################################
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



def rotating(p=.5):
    return Compose([
        ToFloat(),

        ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=45,
                             interpolation=1, border_mode=4, always_apply=False, p=1),

            FromFloat(dtype='uint8', max_value=255.0),

            ], p=p)












