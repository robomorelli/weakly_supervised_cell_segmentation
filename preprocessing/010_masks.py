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

@author: Roberto Morelli
"""
import os
import shutil

import numpy as np
import cv2
import pandas as pd
import sys
sys.path.append('..')
from config import *
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def gaus(x, a, m, s):
    return np.sqrt(a)*np.exp(-(x-m)**2/(2*s**2))
    # if you want it normalized:
    #return 1/(np.sqrt(2*np.pi*s**2))*np.exp(-(x-m)**2/(2*s**2))


def main(args):

	labels_df = pd.read_csv(str(labels_csv).replace('preprocessing', ''))
	img_names = list(set(labels_df['img_name'].values))
	labels_df.set_index(['img_name'], inplace=True)

	if args.mask == 'boolean':

		if Path(str(original_masks).replace('preprocessing', '')).exists():
			shutil.rmtree(str(original_masks).replace('preprocessing', ''))
			os.makedirs(str(original_masks).replace('preprocessing', ''))
		names = []
		for name in img_names:
			mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), np.uint8)
			mask.fill(0)
			print('\nGenerating mask for folder {}\n'.format(name))
			names.append(name)

			try:
				for index, r in labels_df.loc[name].iterrows():
					coords = r['dot']
					coords = [float(x.strip('(),')) for x in coords.split()]
					y, x = coords[0], coords[1]
					if 'RT' in name:
						cv2.circle(mask, (int(x), int(y)), RADIUS_RT, [255, 255, 255], -1)
					else:
						cv2.circle(mask, (int(x), int(y)), 30, [255, 255, 255], -1)
			except:
				print('no coord for ', name)

			mask_name = os.path.join(str(args.saved_masks_path).replace('preprocessing', ''), name)
			plt.imsave(fname=mask_name, arr=mask, cmap='gray')


	else:
		if Path(str(original_masks_gaussian).replace('preprocessing', '')).exists():
			shutil.rmtree(str(original_masks_gaussian).replace('preprocessing', ''))
			os.makedirs(str(original_masks_gaussian).replace('preprocessing', ''))
		RADIUS_RT_ = int(RADIUS_RT*2/3)
		RADIUS_ = int(RADIUS*2/3)
		scale = 255
		limits = 30
		semi_limits=limits//2
		mean = semi_limits
		for name in img_names:
			mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), np.uint8)
			mask.fill(0)
			print('\nGenerating gaussian mask for folder {}\n'.format(name))

			#if len(list(labels_df.loc[name])) > 1:
			try:
				for index, r in list(labels_df.loc[name].iterrows()):
					coords = r['dot']
					if len(coords) > 0:
						coords = [float(x.strip('(),')) for x in coords.split()]
						y, x = int(coords[0]), int(coords[1])
						if 'RT' in name:
							xx, yy = np.meshgrid(np.arange(limits), np.arange(limits))
							gaus2d = gaus(xx, scale, mean, RADIUS_RT_) * gaus(yy, scale, mean, RADIUS_RT_)
							gaus2d_mask = cv2.merge((gaus2d,gaus2d,gaus2d))
							try:
								mask[y-semi_limits:y+semi_limits, x-semi_limits:x+semi_limits] =\
									mask[y-semi_limits:y+semi_limits, x-semi_limits:x+semi_limits] + gaus2d_mask
							except:
								print('cell on boundary')
						else:
							xx, yy = np.meshgrid(np.arange(limits), np.arange(limits))
							gaus2d = gaus(xx, scale, mean, RADIUS_) * gaus(yy, scale, mean, RADIUS_)
							gaus2d_mask = cv2.merge((gaus2d,gaus2d,gaus2d))
							try:
								mask[y-semi_limits:y+semi_limits, x-semi_limits:x+semi_limits] =\
									mask[y-semi_limits:y+semi_limits, x-semi_limits:x+semi_limits] + gaus2d_mask
							except:
								print('cell on boundary')
			except:
				print('no coord for ', name)

			mask = np.clip(mask, 0, 255)
			mask_name = os.path.join(str(args.saved_masks_path).replace('preprocessing', ''), name)
			plt.imsave(fname=mask_name, arr=mask, cmap='gray')

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Define parameters for crop.')
	parser.add_argument('--mask', default='boolean',
						help='boolean or gaussian')
	parser.add_argument('--saved_masks_path', nargs="?", default=original_masks_bigger, help='save masks path')
	args = parser.parse_args()
	main(args)
