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
Created on Fri Jan 18 11:20:02 2019

@author: Marco Dalla
"""
import os
import glob
import tensorflow as tf
from tqdm import tqdm

from config_script import *

def createTFRecord(): 
    
    images_or = glob.glob(str(TRAIN_VALID_OR_PATH)+"/*.TIF")
    images_aug = glob.glob(str(TRAIN_VALID_AUG_OR_PATH)+"/*.TIF")
    images = images_or + images_aug
    images.sort()
    
    masks_or = glob.glob(str(TRAIN_VALID_MASKS_PATH)+"/*.TIF")
    masks_aug = glob.glob(str(TRAIN_VALID_AUG_MASKS_PATH)+"/*.TIF")
    masks = masks_or + masks_aug
    masks.sort()

    # TFRecordWriter, dump to tfrecords file
    if not os.path.exists(os.path.join(str(TRAIN_VALID_OR_PATH.parent), "tfrecords")):
        os.makedirs(os.path.join(str(TRAIN_VALID_OR_PATH.parent), "tfrecords"))
    writer = tf.python_io.TFRecordWriter(os.path.join(str(TRAIN_VALID_OR_PATH.parent), "tfrecords", 
        "images.tfrecords"))
    
    print(len(images))
    
    for image_path, mask_path in tqdm(zip(images,masks),total=len(images)):
        # method to load image.
        image_tf = tf.gfile.FastGFile(image_path, 'rb').read() # image data type is string. 
        # read and binary.
        mask_tf = tf.gfile.FastGFile(mask_path, 'rb').read()
    
        # write bytes to Example proto buffer.
        example = tf.train.Example(features=tf.train.Features(feature={
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_tf])),
            "mask": tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask_tf]))
            }))
        
        writer.write(example.SerializeToString()) # Serialize To String
    
    writer.close()
 
 
 
def readTFRecord(path, BATCH_SIZE):
    
    # read TFRecord
    tf_reader = read_tfrecords.Read_TFRecords(filename=path,
               batch_size=BATCH_SIZE, image_h=IMG_HEIGHT, image_w=IMG_WIDTH,
               image_c=IMG_CHANNELS)

    images, masks = tf_reader.read()
    return image, masks
