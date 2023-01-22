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
    random.seed(args.random_seed)

    if args.color == 'green':
        im_names = os.listdir(train_val_images)
        try:
            shutil.rmtree(str(AugCropImagesFewShot) + '_{}_{}'.format(args.few_shot, args.random_seed))
            os.makedirs(str(FewShot) + '/images_{}_{}'.format(args.few_shot, args.random_seed), exist_ok=True)
        except:
            pass
        os.makedirs(str(AugCropImagesFewShot) + '_{}_{}'.format(args.few_shot, args.random_seed), exist_ok=True)
        os.makedirs(str(FewShot) + '/images_{}_{}'.format(args.few_shot, args.random_seed), exist_ok=True)
        full_images_path = str(FewShot) + '/images_{}_{}'.format(args.few_shot, args.random_seed)
        try:
            shutil.rmtree(str(AugCropMasksFewShot) + '_{}_{}'.format(args.few_shot, args.random_seed))
            os.makedirs(str(FewShot) + '/masks_{}_{}'.format(args.few_shot, args.random_seed), exist_ok=True)
        except:
            pass
        os.makedirs(str(AugCropMasksFewShot) + '_{}_{}'.format(args.few_shot,args.random_seed), exist_ok=True)
        os.makedirs(str(FewShot) + '/masks_{}_{}'.format(args.few_shot, args.random_seed), exist_ok=True)
        full_masks_path = str(FewShot) + '/masks_{}_{}'.format(args.few_shot, args.random_seed)


        if args.few_shot != 'all':
            #indexes = np.random.randint(0, len(im_names), args.few_shot)
            indexes = random.sample(range(len(im_names)), args.few_shot)
        else:
            indexes = np.arange(0, len(im_names))

        to_sample_root = [im_names[i] for i in indexes]

        cropped_names = os.listdir(aug_cropped_train_val_images)
        copied = []
        for im_names in cropped_names:
            name = '_'.join(im_names.split('_')[:-2]) + '.png'

            if name in to_sample_root:

                shutil.copy(os.path.join(aug_cropped_train_val_images,im_names), os.path.join(str(AugCropImagesFewShot)\
                                                                                              + '_{}_{}'.format(args.few_shot, args.random_seed), im_names))
                shutil.copy(os.path.join(aug_cropped_train_val_masks,im_names), os.path.join(str(AugCropMasksFewShot)\
                                                                                             + '_{}_{}'.format(args.few_shot, args.random_seed), im_names))

                shutil.copy(os.path.join(train_val_images,name), os.path.join(full_images_path, name))
                shutil.copy(os.path.join(train_val_masks,name), os.path.join(full_masks_path, name))
                copied.append(name)
        print(np.unique(copied))


    else:
        raise NotImplementedError


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Define parameters for test.')
    parser.add_argument('--color', nargs="?", default='green', help='the folder including the masks to crop')
    parser.add_argument('--few_shot', nargs="?", default=200, help='the folder including the masks to crop')
    parser.add_argument('--random_seed', nargs="?", default=123, help='the folder including the masks to crop')
    args = parser.parse_args()

    main(args)
