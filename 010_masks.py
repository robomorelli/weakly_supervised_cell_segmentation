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
Created on Thu Jan 10 10:58:10 2019

@author: Luca Clissa
"""
import os
import sys
from pathlib import Path

CODE_DIRECTORY = Path.home() / "project/code/sample"
sys.path.append(str(CODE_DIRECTORY))
from config_script import *

import random
import numpy as np
import matplotlib.pyplot as plt

import re
from tqdm import tqdm

import itertools
from itertools import chain

import imageio
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.color import rgb2gray
import cv2

x_fact = 1600/IMG_WIDTH
y_fact = 1200/IMG_HEIGHT

for counter, sample in enumerate(COORDINATE_PATH):
#    print(counter, sample)
    print('\nGenerating mask for folder {}\n'.format(sample.parent.name))
	
    if 'RT' in sample.parent.name:

	    for file_path in sample.iterdir():
	    #        print(file_path)
		    number = re.findall(r'[0-9]+',file_path.name)[0]
		    file = open(file_path,'r')
		    coordinates = []
		    newCoord = []
		    for line in file:
			    if line.replace(' ','').startswith('P'):
				    continue
			    coordinates.append(line.split('\t')[2:4])
			    numeric_coordinates = [[int(j) for j in i] for i in coordinates]
			    newCoord = [(int(x/x_fact), int(y/y_fact)) for x, y in numeric_coordinates]
		    img = np.zeros([IMG_HEIGHT,IMG_WIDTH,3],dtype=np.uint8)
		    img.fill(0)
		    for pt in newCoord:
			    #print(pt[0],pt[1])
			    cv2.circle(img, (pt[0], pt[1]), 6, [255,255,255], -1)
		    if not len(newCoord):
			    print("\nNo coordinates in file: ", file_path)
		    output_path = MASKS_PATH[counter] / 'campione{}.TIF'.format(number)       
		    imageio.imwrite( str(output_path), img)
				    #print('writing campione_%d of folder %s', (number,fol))
		    file.close()
			
    else:
	
		
	    for file_path in sample.iterdir():
	    #        print(file_path)
		    number = re.findall(r'[0-9]+',file_path.name)[0]
		    file = open(file_path,'r')
		    coordinates = []
		    newCoord = []
		    for line in file:
			    if line.replace(' ','').startswith('P'):
				    continue
			    coordinates.append(line.split('\t')[2:4])
			    numeric_coordinates = [[int(j) for j in i] for i in coordinates]
			    newCoord = [(int(x/x_fact), int(y/y_fact)) for x, y in numeric_coordinates]
		    img = np.zeros([IMG_HEIGHT,IMG_WIDTH,3],dtype=np.uint8)
		    img.fill(0)
		    for pt in newCoord:
			    #print(pt[0],pt[1])
			    cv2.circle(img, (pt[0], pt[1]), RADIUS, [255,255,255], -1)
		    if not len(newCoord):
			    print("\nNo coordinates in file: ", file_path)
		    output_path = MASKS_PATH[counter] / 'campione{}.TIF'.format(number)       
		    imageio.imwrite( str(output_path), img)
				    #print('writing campione_%d of folder %s', (number,fol))
		    file.close()
	