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
from skimage.morphology import watershed, remove_small_holes, remove_small_objects,\
label, erosion, dilation, local_maxima, skeletonize
from scipy import ndimage
from skimage.feature import peak_local_max

reader = import_module('020_loader')
from config import *

LoadImagesForWeight = ALL_IMAGES # Image without Test
LoadMasksForWeight = ALL_MASKS #Need to rerun mask generator for create mask and relative path

SaveWeightsMasksRed =  str(ALL_MASKS) #Create this folder in config (unique folder)

#Read all images ids in all images path, they could be also 
#all images after augumentation (10K images and more)
image_ids_name = check_output(["ls", LoadMasksForWeight]).decode("utf8").split()

#To make weights needed to take the only first 2064 coming from cropper (cropped original images)
#Better if they are sorted by num
# ix = [x for x in range(2064)]
ix = [x for x in range(2964)] #new red images with 19 images in test
ix.sort()
image_ids = [str(x)+'.TIF' for x in ix]

sigma = 4

def make_weights(image_ids, SaveWeightsMasksRed = SaveWeightsMasksRed, maximum = 6.224407):

    maximum = maximum
    
    for ax_index, name in tqdm(enumerate(image_ids),total=len(image_ids)):

        image, target = reader.loadImage(name, 'train', 'all_images', True)
        
        target = target[:,:,0:1]
        
        target_ero = erosion(np.squeeze(target), selem=np.ones([3,3]))

        tar_inv = cv2.bitwise_not(target)
        tar_dil = dilation(np.squeeze(target), selem=np.ones([21, 21]))

        mask_sum = cv2.bitwise_and(tar_dil, tar_inv)
        # mask_sum1 = cv2.bitwise_or(mask_sum, target_ero)
        mask_sum1 = cv2.bitwise_or(mask_sum, target)

        edge_strip = cv2.subtract(target, target_ero)
        edge_strip = np.clip(edge_strip, 0, 1).astype(np.float32)

        null = np.zeros((target.shape[0], target.shape[1]), dtype = np.float32)
        weighted_mask = np.zeros((target.shape[0], target.shape[1]), dtype = np.float32)

        mask, nlabels_mask = ndimage.label(target)
        
        if nlabels_mask < 1:
                       
            weighted_maskkk = np.ones((target.shape[0], target.shape[1]), dtype = np.float32)
            
        else:
        
        
            mask = remove_small_objects(mask, min_size=25, connectivity=1, in_place=False)
            mask, nlabels_mask = ndimage.label(mask)
            mask_objs = ndimage.find_objects(mask)

            edge_pre = ndimage.distance_transform_edt(target_ero==0)
            
            edge = (1.4*np.exp((-1 * (edge_pre) ** 2) / (2 * (sigma ** 2)))).astype(np.float32)


            for idx,obj in enumerate(mask_objs):
#                     print('obj {} on {}'.format(idx+1, nlabels_mask))
                new_image = np.zeros_like(mask)
                new_image[obj[0].start:obj[0].stop,obj[1].start:obj[1].stop] = mask[obj]  

                new_image = np.clip(new_image, 0, 1).astype(np.uint8)
                new_image *= 255

                inverted = cv2.bitwise_not(new_image)

                distance = ndimage.distance_transform_edt(inverted)
                w = np.zeros((distance.shape[0],distance.shape[1]), dtype=np.float32)
                w1 = np.zeros((distance.shape[0],distance.shape[1]), dtype=np.float32)

                for i in range(distance.shape[0]):

                    for j in range(distance.shape[1]):

                        if distance[i, j] != 0:

                            w[i, j] = 1.*np.exp((-1 * (distance[i,j]) ** 2) / (2 * (19 ** 2)))

                        else:

                            w[i, j] = 1


                weighted_mask = cv2.add(weighted_mask, w, mask = tar_inv)

            # Complete from inner to edge with 1.5 as weight 
            weighted_mask = np.clip(weighted_mask, 1, weighted_mask.max())

            mul = target_ero*1.5/255
            mul = mul.astype(np.float32)
            mul = np.clip(mul,1,mul.max())

            board = cv2.multiply(edge, edge_strip)
            board = np.clip(board,1,mul.max())
            weighted_maskk = cv2.multiply(weighted_mask, mul)

            weighted_maskkk = cv2.multiply(weighted_maskk, board)
            
            
#             total[ax_index] = weighted_maskkk
            
            
        if (weighted_maskkk.max()/(maximum+0.0001))> 1:

            break

        weighted_maskkk = weighted_maskkk*1/maximum
        target = np.clip(target, 0 , 1)
        final_target = np.dstack((target, weighted_maskkk, null))


        aug_mask_dir =  SaveWeightsMasksRed + '/{}.TIF'.format(ax_index) 
        
        # print('saving {}'.format(name))
        plt.imsave(fname=aug_mask_dir,arr = final_target)
            
make_weights(image_ids, SaveWeightsMasksRed, maximum = 6.224407)


















