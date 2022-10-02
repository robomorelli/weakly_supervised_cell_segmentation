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
Created on Fri Jan 11 12:10:10 2019

@author: Marco Dalla
"""
import glob
import random
import matplotlib.pyplot as plt
import cv2
import os
import re
import pickle
import random
from shutil import copyfile, move
from skimage.transform import resize
from tqdm import tqdm


from config_script import *

IMG_CHANNELS = 3

def moveImages():
    
    final_dict = {}
    idx = 0
    folders = glob.glob(str(DATA_DIRECTORY)+'/'+'Mar*')+glob.glob(str(DATA_DIRECTORY)+'/'+'RT*')
    
    for folder in folders:
        dict0 = {}
        os.chdir(folder)
        names = open('nomi.txt','r')
        lines = names.readlines()
        for line in lines:
            line = line.replace('\n','')
            dict0[line.split('\t')[2]]=line.split('\t')[0]
        or_images = glob.glob(folder+'/original_images/'+'*.TIF')
        masks = glob.glob(folder+'/'+TRAIN_VALID_MASKS_PATH.name+'/*.TIF')
        or_images.sort()
        masks.sort()
        
        for (image,mask) in tqdm(zip(or_images,masks),total=len(or_images)):
            im_name = image.split('/')[-1]
            im_name = im_name.replace('.TIF','')
            fin_name = str(idx) #(re.findall(r'\d+', im_name)[0])
            final_name= str(idx)+'.TIF'
            img = cv2.imread(image)
            img_resized = resize(img, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant')
            plt.imsave(fname=str(TRAIN_VALID_OR_PATH)+'/'+final_name, arr=img_resized)
            copyfile(mask,str(TRAIN_VALID_MASKS_PATH)+'/'+final_name)
            final_dict[fin_name] = dict0[im_name]
            idx += 1
        
    #print(final_dict)
    with open(str(PROJECT_DIRECTORY)+'/map.pkl','wb') as fp:
        pickle.dump(final_dict, fp)
    with open(str(PROJECT_DIRECTORY)+'/map.txt', 'w') as f:
        print(final_dict, file=f)
        train_original_images = os.listdir(str(TRAIN_VALID_OR_PATH))
    test_images = random.sample(train_original_images,TEST_IMG)
    for image in tqdm(test_images, total=len(test_images)):
        move(str(TRAIN_VALID_OR_PATH)+'/'+image,str(TEST_OR_PATH)+'/'+image)
        move(str(TRAIN_VALID_MASKS_PATH)+'/'+image,str(TEST_MASKS_PATH)+'/'+image)
    
    
def loadImage(image_id, train_test='train', or_crop='crop', labels=True):
    
    if train_test == 'train' and or_crop == 'original':
        or_images_path = str(TRAIN_VALID_OR_PATH) + '/'
        crop_masks_path = str(TRAIN_VALID_MASKS_PATH) + '/'
    elif train_test == 'train' and or_crop == 'crop':    
        crop_images_path = str(TRAIN_VALID_CROP_OR_PATH) + '/'
        crop_masks_path = str(TRAIN_VALID_CROP_MASKS_PATH) + '/'
    elif train_test == 'test':
        or_images_path = str(TEST_OR_PATH) + '/'
        or_masks_path = str(TEST_MASKS_PATH) + '/'
    elif train_test == 'train' and or_crop == 'all_images':    
        images_path = str(ALL_IMAGES) + '/'
        masks_path = str(ALL_MASKS) + '/'
        

    if or_crop == 'original' and labels == True:
        img_x = cv2.imread( or_images_path + image_id)
        img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)
        mask = cv2.imread( crop_masks_path + image_id)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        return img_x, mask
    elif or_crop == 'original' and labels == False:
        img_x = cv2.imread( or_images_path + image_id)
        return img_x
    elif or_crop == 'crop' and labels == True:
        img_x = cv2.imread( crop_images_path + image_id)
        mask = cv2.imread( crop_masks_path + image_id)
        return img_x, mask
    elif or_crop == 'all_images' and labels == True:
        img_x = cv2.imread(images_path + image_id)
        img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(masks_path + image_id)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        return img_x, mask
    elif or_crop == 'crop' and labels == False:
        img_x = cv2.imread( crop_images_path + image_id)
        return img_x
    
def loadGenerator(path):
    img = cv2.imread(path)
    return img
    
def generator(images, labels, batch_size):

    batch_img_x = np.zeros((batch_size, 512, 512, 3))
    batch_img_y = np.zeros((batch_size, 512, 512, 3))

    while True:
        for i in range(batch_size):
            # choose random index in features
            index= random.choice(len(images),1)
            batch_img_x[i] = loadGenerator(images[index])
            batch_img_y[i] = loadGenerator(labels[index])
        yield batch_img_x,batch_img_y

if __name__ == "__main__":
    moveImages()