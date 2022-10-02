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

@author: Luca Clissa
"""

import os
import sys
from pathlib import Path, PureWindowsPath #needed for displaying Windows paths

# NEEDED FOR cv2 ENVIRONMENT (ONLY LUCA)
import re
import getpass
if re.match('luca',getpass.getuser()):
    tf_path = "/home/luca/Downloads/yes/envs/tensorflow/lib/python3.6/site-packages"
    sys.path.append(tf_path)
import cv2

__all__ = [
        'IMG_SIZE',
        'IMG_WIDTH',
        'IMG_HEIGHT',
        'IMG_CHANNELS',
        'RADIUS',
        'PROJECT_DIRECTORY',
        'DATA_DIRECTORY',
        'CODE_DIRECTORY',
        'SAMPLE_FOLDERS',
        'TRAIN_VALID_OR_PATH',
        'TRAIN_VALID_MASKS_PATH',
        'TRAIN_VALID_AUG_OR_PATH',
        'TRAIN_VALID_AUG_MASKS_PATH',
        'TEST_OR_PATH',
        'TEST_MASKS_PATH',
        'ORIGINAL_IMG_PATH',
        'COORDINATE_PATH',
        # 'MASKS_PATH',
        'count_files_in_directory',
        'RESULTS_DIRECTORY',
        'MODEL_CHECKPOINTS',
        'TRAIN_VALID_CROP_OR_PATH',
        'TRAIN_VALID_CROP_MASKS_PATH',
        'ALL_IMAGES',
        'ALL_MASKS',
        'TEST_CROP_MASKS_PATH',
        'TEST_CROP_OR_PATH',
        'PRED_IMAGES'
        ]

home_path = Path.home()
PROJECT_DIRECTORY = Path("/home/imagepro") / "project"
DATA_DIRECTORY = PROJECT_DIRECTORY / "data/raw_data"

CODE_DIRECTORY = PROJECT_DIRECTORY / "code/sample"
RESULTS_DIRECTORY = PROJECT_DIRECTORY / "results"
MODEL_CHECKPOINTS = RESULTS_DIRECTORY / "model_checkpoints"
MODEL_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

if str(CODE_DIRECTORY) not in sys.path:
    sys.path.append(str(CODE_DIRECTORY))

# set image dimensions
IMG_SIZE = 512
IMG_WIDTH = 1600
IMG_HEIGHT = 1200
IMG_CHANNELS = 1

# set mask's dots radius
RADIUS = 12

# set augmentation factor
SPLIT_NUM = 4


# set some important paths


if (IMG_SIZE == IMG_WIDTH & IMG_SIZE == IMG_HEIGHT):
    MASKS_FOLDER = "masks_" + str(IMG_SIZE) + "_" + str(RADIUS)    
    AUG_MASKS_FOLDER = "aug_masks_" + str(IMG_SIZE) + "_" + str(RADIUS)
else:
    MASKS_FOLDER = ("masks_" + str(IMG_WIDTH) + "_" + str(IMG_HEIGHT) 
    + "_" + str(RADIUS))
    AUG_MASKS_FOLDER = ("aug_masks_" + str(IMG_WIDTH) + "_" 
    + str(IMG_HEIGHT) + "_" + str(RADIUS))
    
#path_on_windows = PureWindowsPath(PROJECT_DIRECTORY)
#print(path_on_windows)

# training and validation
TRAIN_VALID_OR_PATH = PROJECT_DIRECTORY / "data/train_valid" / "original_images"
TRAIN_VALID_MASKS_PATH = PROJECT_DIRECTORY / "data/train_valid" / MASKS_FOLDER
TRAIN_VALID_CROP_OR_PATH = PROJECT_DIRECTORY / "data/train_valid" / "cropped_images"
TRAIN_VALID_CROP_MASKS_PATH = PROJECT_DIRECTORY / "data/train_valid" / "cropped_masks"
TRAIN_VALID_AUG_OR_PATH = PROJECT_DIRECTORY / "data/train_valid" / "aug_images"
TRAIN_VALID_AUG_MASKS_PATH = PROJECT_DIRECTORY / "data/train_valid" / AUG_MASKS_FOLDER
ALL_IMAGES = PROJECT_DIRECTORY / "data/train_valid" / "all_images" / "images"
ALL_MASKS = PROJECT_DIRECTORY / "data/train_valid" / "all_masks" / "masks"
PRED_IMAGES = PROJECT_DIRECTORY / "data/train_valid" / "predicted" / "images"

# test
TEST_CROP_MASKS_PATH = PROJECT_DIRECTORY / "data/test" / "cropped_images"
TEST_CROP_OR_PATH = PROJECT_DIRECTORY / "data/test" / "cropped_masks"
TEST_OR_PATH = PROJECT_DIRECTORY / "data/test" / "original_images" / "images"
TEST_MASKS_PATH = PROJECT_DIRECTORY / "data/test" / MASKS_FOLDER / "masks"

#print(TRAIN_VALID_MASKS_PATH, TRAIN_VALID_OR_PATH)
# initialize lists for original images, coordinates and masks path
try:
    os.chdir(DATA_DIRECTORY)
except FileNotFoundError:
    print("WARNING: either data folder is not present or its location is different than expected")
SAMPLE_FOLDERS = os.popen("ls").read().split("\n")[:-1]


ORIGINAL_IMG_PATH = [ DATA_DIRECTORY / sample / "original_images" 
                     for sample in SAMPLE_FOLDERS ]
COORDINATE_PATH = [ DATA_DIRECTORY / sample / "coordinates" 
                     for sample in SAMPLE_FOLDERS ]
MASKS_PATH = [ DATA_DIRECTORY / sample / MASKS_FOLDER
                     for sample in SAMPLE_FOLDERS ]


if len(sys.argv) > 1:
    print("Original paths:\n\n", ORIGINAL_IMG_PATH)
    print("\nCoordinates:\n\n", COORDINATE_PATH)
    print("\nMasks paths:\n\n", MASKS_PATH)

# create folders needed for following modules
TRAIN_VALID_OR_PATH.mkdir(parents=True, exist_ok=True)
TRAIN_VALID_MASKS_PATH.mkdir(parents=True, exist_ok=True)
TRAIN_VALID_CROP_OR_PATH.mkdir(parents=True, exist_ok=True)
TRAIN_VALID_CROP_MASKS_PATH.mkdir(parents=True, exist_ok=True)
TRAIN_VALID_AUG_OR_PATH.mkdir(parents=True, exist_ok=True)
TRAIN_VALID_AUG_MASKS_PATH.mkdir(parents=True, exist_ok=True)
TEST_OR_PATH.mkdir(parents=True, exist_ok=True)
TEST_MASKS_PATH.mkdir(parents=True, exist_ok=True)
TEST_CROP_OR_PATH.mkdir(parents=True, exist_ok=True)
TEST_CROP_MASKS_PATH.mkdir(parents=True, exist_ok=True)
ALL_IMAGES.mkdir(parents=True, exist_ok=True)
ALL_MASKS.mkdir(parents=True, exist_ok=True)
PRED_IMAGES.mkdir(parents=True, exist_ok=True)

for masks_path in MASKS_PATH:
    masks_path.mkdir(parents=True, exist_ok=True)

def count_files_in_directory(count_start=0, dir_list=ORIGINAL_IMG_PATH):
    if type(dir_list) != type([]):
        dir_list = [dir_list]
    tot_files = count_start
    for directory in dir_list:
        tot_files += sum(1 for _ in directory.iterdir() if _.is_file())
    return tot_files

#try:
#    TOT_IMG = count_files_in_directory()
#    TEST_IMG = int(round(TOT_IMG*0.10,0))
#except FileNotFoundError:
#    print("TOT_IMG and TEST_IMG cannot be initialized since there is no data directory")

