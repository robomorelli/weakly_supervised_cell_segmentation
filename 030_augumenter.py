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

from config_script import *

augmenter = import_module('031_augumenter_utils')
reader = import_module('020_loader')

#image_ids = check_output(["ls", TRAIN_VALID_OR_PATH]).decode("utf8").split()
split_num = 5

#On previous version of make data augumentation, among the arguments of the function were also: shift and image_ids

augmenter.make_augumentation_on_red(split_num, ALL_IMAGES, ALL_MASKS, start = 'begin')