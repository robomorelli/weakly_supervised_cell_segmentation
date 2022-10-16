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

import argparse
from config import *
from preprocessing.utils import make_data_augmentation
import sys
import shutil
sys.path.append('..')


def main(args):

    make_data_augmentation(args.split_num, args.images_path, args.masks_path,
                           args.saved_images_path, args.saved_masks_path)

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Define parameters for crop.')

    parser.add_argument('--split_num', default=5,
                        help='augumenteation factor')

    parser.add_argument('--images_path', nargs="?", default=cropped_train_val_images,
                        help='the folder including the images to crop')
    parser.add_argument('--masks_path', nargs="?", default=cropped_train_val_masks, help='the folder including the masks to crop')
    parser.add_argument('--saved_images_path', nargs="?", default=aug_cropped_train_val_images, help='save images path')
    parser.add_argument('--saved_masks_path', nargs="?", default=aug_cropped_train_val_masks, help='save masks path')
    parser.add_argument('--elastic', action='store_const', const=True, default=False)
    parser.add_argument('--cropped', action='store_const', const=True, default=True)
    parser.add_argument('--start_from_zero', action='store_const', const=True, default=True,
                        help='remove previous file in the destination folder')

    args = parser.parse_args()

    if args.start_from_zero:
        print('deleting existing files in destination folder')
        try:
            shutil.rmtree(args.saved_images_path)
            os.makedirs(args.saved_images_path, exist_ok=True)
        except:
            pass

        try:
            shutil.rmtree(args.saved_masks_path)
            os.makedirs(args.saved_masks_path, exist_ok=True)
        except:
            pass

    main(args)
