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
from pathlib import Path

def main():

	labels_df = pd.read_csv(str(labels_csv).replace('preprocessing', ''))
	img_names = labels_df['img_name'].values
	labels_df.set_index(['img_name'], inplace=True)

	if Path(str(original_masks).replace('preprocessing', '')).exists():
		shutil.rmtree(str(original_masks).replace('preprocessing', ''))
		os.makedirs(str(original_masks).replace('preprocessing', ''))

	for name in img_names:
		mask = np.zeros((IMG_HEIGHT,IMG_WIDTH,3), np.uint8)
		mask.fill(0)
		print('\nGenerating mask for folder {}\n'.format(name))

		try:
			for index, r in labels_df.loc[name].iterrows():
				coords = r['dot']
				coords = [float(x.strip('(),')) for x in coords.split()]
				y, x = coords[0], coords[1]
				if 'RT' in name:
					cv2.circle(mask, (int(x), int(y)), RADIUS_RT, [255, 255, 255], -1)
				else:
					cv2.circle(mask, (int(x), int(y)), RADIUS, [255, 255, 255], -1)
		except:
			print('no coord for ',name)

		mask_name = os.path.join(str(original_masks).replace('preprocessing', ''), name)
		plt.imsave(fname=mask_name, arr=mask, cmap='gray')

if __name__ == "__main__":
	main()
