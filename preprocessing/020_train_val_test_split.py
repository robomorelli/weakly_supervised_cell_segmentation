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
import shutil
import sys
import numpy as np
from tqdm import tqdm
from subprocess import check_output
import matplotlib.pyplot as plt
sys.path.append('..')
from utils import read_image_masks, read_masks, read_images
from config import *
import random

LoadImagesForSplit = str(original_images).replace('preprocessing', '')
LoadMasksForSplit = str(original_masks).replace('preprocessing', '')

SaveTrainValImages = str(train_val_images).replace('preprocessing', '')
SaveTrainValMasks = str(train_val_masks).replace('preprocessing', '')

SaveTestImages = str(test_images).replace('preprocessing', '')
SaveTestMasks = str(test_masks).replace('preprocessing', '')

path_list = [SaveTrainValImages, SaveTrainValMasks, SaveTestImages, SaveTestMasks]
random.seed(10)

def main(train_val_split):

    images_name = check_output(["ls", LoadImagesForSplit]).decode("utf8").split()
    tot_images = len(images_name)
    train_val_len = int(train_val_split*tot_images)

    train_val_indices = random.sample(range(0,tot_images), train_val_len)
    train_val_names = [images_name[i] for i in train_val_indices]

    for path in path_list:
        if Path(str(path).replace('preprocessing/', '')).exists():
            shutil.rmtree(str(path).replace('preprocessing/', ''))
            os.makedirs(str(path).replace('preprocessing/', ''))

    for ix, name in tqdm(enumerate(images_name), total=len(images_name)):
        if name in train_val_names:
            shutil.copy(os.path.join(LoadImagesForSplit,name), os.path.join(SaveTrainValImages,name))
            shutil.copy(os.path.join(LoadMasksForSplit,name), os.path.join(SaveTrainValMasks,name))
        else:
            shutil.copy(os.path.join(LoadImagesForSplit,name), os.path.join(SaveTestImages,name))
            shutil.copy(os.path.join(LoadMasksForSplit,name), os.path.join(SaveTestMasks,name))
    return

if __name__ == "__main__":
    train_val_split = 0.85
    main(train_val_split)


