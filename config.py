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
from pathlib import Path

root = os.getcwd()

# set image dimensions
IMG_SIZE = 512
IMG_WIDTH = 1600
IMG_HEIGHT = 1200
IMG_CHANNELS = 3
# set mask's dots radius
RADIUS = 12
RADIUS_RT = 6
# set augmentation factor
SPLIT_NUM = 4

home_path = os.getcwd()
home_path  = Path(home_path).as_posix()

if home_path .split('/')[-1] != "weakly_supervised_cell_segmentation":
    home_path = Path(home_path ).parent.as_posix()

PROJECT_DIRECTORY = Path(home_path)

model_results = PROJECT_DIRECTORY / "model_results"
model_results_supervised_green = PROJECT_DIRECTORY / "model_results" / "supervised/green"
model_results_supervised_yellow = PROJECT_DIRECTORY / "model_results" / "supervised/yellow"

# training and validation
labels_csv = PROJECT_DIRECTORY / "data/labels.csv"
original_images = PROJECT_DIRECTORY / "data" / "original/images"
original_masks = PROJECT_DIRECTORY / "data" / "original/masks"
original_masks_bigger = PROJECT_DIRECTORY / "data" / "original/masks_bigger"
original_masks_gaussian = PROJECT_DIRECTORY / "data" / "original/masks_gaussian"

train_val_images = PROJECT_DIRECTORY / "data/train_val" / "images"
train_val_masks = PROJECT_DIRECTORY / "data/train_val" / "masks"
train_val_masks_bigger = PROJECT_DIRECTORY / "data/train_val" / "bigger_masks"
train_val_ws_masks = PROJECT_DIRECTORY / "data/train_val" / "weakly_supervised_masks"
train_val_gauss_masks = PROJECT_DIRECTORY / "data/train_val" / "gaussian_masks"

cropped_train_val_images = PROJECT_DIRECTORY / "data/train_val/cropped" / "images"
cropped_train_val_masks = PROJECT_DIRECTORY / "data/train_val/cropped" / "masks"
cropped_train_val_ws_images = PROJECT_DIRECTORY / "data/train_val/cropped" / "weakly_supervised_images"
cropped_train_val_ws_masks = PROJECT_DIRECTORY / "data/train_val/cropped" / "weakly_supervised_masks"
cropped_train_val_gauss_images = PROJECT_DIRECTORY / "data/train_val/cropped" / "gaussian_images"
cropped_train_val_gauss_masks = PROJECT_DIRECTORY / "data/train_val/cropped" / "gaussian_masks"

aug_cropped_train_val_images = PROJECT_DIRECTORY / "data/train_val/aug_cropped" / "images"
aug_cropped_train_val_masks = PROJECT_DIRECTORY / "data/train_val/aug_cropped" / "masks"

aug_cropped_train_val_images_massive = PROJECT_DIRECTORY / "data/train_val/aug_cropped" / "images_massive"
aug_cropped_train_val_masks_massive = PROJECT_DIRECTORY / "data/train_val/aug_cropped" / "masks_massive"

aug_cropped_train_val_ws_images = PROJECT_DIRECTORY / "data/train_val/aug_cropped" / "weakly_supervised_images"
aug_cropped_train_val_ws_masks = PROJECT_DIRECTORY / "data/train_val/aug_cropped" / "weakly_supervised_masks"

aug_cropped_train_val_gauss_images = PROJECT_DIRECTORY / "data/train_val/aug_cropped" / "gaussian_images"
aug_cropped_train_val_gauss_masks = PROJECT_DIRECTORY / "data/train_val/aug_cropped" / "gaussian_masks"

test_images = PROJECT_DIRECTORY / "data/test" / "original_images"
test_masks = PROJECT_DIRECTORY / "data/test" / "original_masks"

AugCropImagesFewShot = PROJECT_DIRECTORY / "data/train_val/aug_cropped_few_shot" / "images"
AugCropMasksFewShot = PROJECT_DIRECTORY / "data/train_val/aug_cropped_few_shot" / "masks"

FewShot = PROJECT_DIRECTORY / "data/train_val/few_shot"
FewShot = PROJECT_DIRECTORY / "data/train_val/few_shot"

# create folders needed for following modules
if 'preprocessing' not in str(original_images):
    original_images.mkdir(parents=True, exist_ok=True)
    original_masks.mkdir(parents=True, exist_ok=True)
    original_masks_bigger.mkdir(parents=True, exist_ok=True)
    train_val_masks_bigger.mkdir(parents=True, exist_ok=True)
    train_val_images.mkdir(parents=True, exist_ok=True)
    train_val_masks.mkdir(parents=True, exist_ok=True)
    train_val_ws_masks.mkdir(parents=True, exist_ok=True)
    train_val_gauss_masks.mkdir(parents=True, exist_ok=True)

    cropped_train_val_images.mkdir(parents=True, exist_ok=True)
    cropped_train_val_masks.mkdir(parents=True, exist_ok=True)
    cropped_train_val_ws_images.mkdir(parents=True, exist_ok=True)
    cropped_train_val_ws_masks.mkdir(parents=True, exist_ok=True)
    cropped_train_val_gauss_images.mkdir(parents=True, exist_ok=True)
    cropped_train_val_gauss_masks.mkdir(parents=True, exist_ok=True)

    aug_cropped_train_val_images.mkdir(parents=True, exist_ok=True)
    aug_cropped_train_val_masks.mkdir(parents=True, exist_ok=True)
    aug_cropped_train_val_ws_images.mkdir(parents=True, exist_ok=True)
    aug_cropped_train_val_ws_masks.mkdir(parents=True, exist_ok=True)
    aug_cropped_train_val_gauss_images.mkdir(parents=True, exist_ok=True)
    aug_cropped_train_val_gauss_masks.mkdir(parents=True, exist_ok=True)

    aug_cropped_train_val_images_massive.mkdir(parents=True, exist_ok=True)
    aug_cropped_train_val_masks_massive.mkdir(parents=True, exist_ok=True)

    test_images.mkdir(parents=True, exist_ok=True)
    test_masks.mkdir(parents=True, exist_ok=True)
    AugCropImagesFewShot.mkdir(parents=True, exist_ok=True)
    AugCropMasksFewShot.mkdir(parents=True, exist_ok=True)
    original_masks_gaussian.mkdir(parents=True, exist_ok=True)

    model_results.mkdir(parents=True, exist_ok=True)

