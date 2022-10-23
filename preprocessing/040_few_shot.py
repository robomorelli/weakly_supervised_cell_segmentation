import os

import pandas as pd
import numpy as np
import argparse
import os
import shutil
import random
from config import *

np.random.seed(0)

def main(args):

    if args.color == 'green':
        im_names = os.listdir(train_val_images)
        try:
            shutil.rmtree(AugCropImagesFewShot)
        except:
            pass
        os.makedirs(AugCropImagesFewShot, exist_ok=True)
        try:
            shutil.rmtree(AugCropMasksFewShot)
        except:
            pass
        os.makedirs(AugCropMasksFewShot, exist_ok=True)

        if args.few_shot != 'all':
            indexes = np.random.randint(0, len(im_names), args.few_shot)
        else:
            indexes = np.arange(0, len(im_names))

        to_sample_root = [im_names[i] for i in indexes]

        cropped_names = os.listdir(aug_cropped_train_val_images)
        for im_names in cropped_names:
            if '_'.join(im_names.split('_')[:-2]) + '.png' in to_sample_root:
                shutil.copy(os.path.join(aug_cropped_train_val_images,im_names), os.path.join(AugCropImagesFewShot, im_names))
                shutil.copy(os.path.join(aug_cropped_train_val_masks,im_names), os.path.join(AugCropMasksFewShot, im_names))

    else:
        raise NotImplementedError


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Define parameters for test.')
    parser.add_argument('--color', nargs="?", default='green', help='the folder including the masks to crop')
    parser.add_argument('--few_shot', nargs="?", default=10, help='the folder including the masks to crop')
    args = parser.parse_args()

    main(args)
