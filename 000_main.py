#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:35:40 2019

@author: Luca Clissa
"""
import os
import sys
from pathlib import Path
import importlib

# import custome modules
CODE_DIRECTORY = Path.home() / "project/code"
sys.path.append(str(CODE_DIRECTORY))
from config_script import *
loader = importlib.import_module('020_loader')
augmenter = importlib.import_module('031_augumenter')

# move original images and masks in a unique directory 
loader.moveImages()

image_ids = augmenter.check_output(["ls", TRAIN_VALID_OR_PATH]).decode("utf8").split()
augmenter.make_data_augmentation(image_ids,SPLIT_NUM, TOT_IMG,
                                 TRAIN_VALID_AUG_OR_PATH, TRAIN_VALID_AUG_MASKS_PATH )

