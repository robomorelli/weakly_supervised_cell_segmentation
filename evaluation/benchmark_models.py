#IMPORT LIBRARIES
import sys
from pathlib import Path
import glob
import os
from tqdm import tqdm
from shutil import copyfile
import pickle

import numpy as np
import cv2

from scipy import ndimage
import skimage
from skimage.transform import resize
from skimage.morphology import label
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
from skimage.morphology import remove_small_holes, remove_small_objects, label
from skimage.feature import peak_local_max

from skimage.segmentation import watershed

sys.path.append('..')
sys.path.append('../dataset_loader')
sys.path.append('../model')

from dataset_loader.image_loader import *
from model.resunet import *
from utils import *
from torch.utils.data import DataLoader

from config import *

import torch
from torchvision.transforms import transforms as T
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
import argparse
from kneed import KneeLocator
matplotlib.use('qt5Agg')

from evaluation.evaluation_utils import post_processing, compute_metrics_global, model_inference, compute_metrics

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

def metric_reports_th_sweep(args, val_images_path, val_masks_path, test_images_path, test_masks_path,
                            model_path, save_metrics_path,
                            batch_size=4, c0=True, transform=None, th_min=0.3, th_max=1,
                            steps=5, min_obj_size=2, foot=4, area_threshold=6, max_dist=3):
    # load model
    vae_flag = False
    if args.architecture == 'vae':
        model = nn.DataParallel(
            c_resunetVAE(arch='c-ResUnetVAE', n_features_start=16, n_out=1, n_outRec=3, fully_conv=False,
                         pretrained=False, progress=True)).to(device)
        model.load_state_dict(torch.load(model_path))
        vae_flag = True
    elif args.architecture == 'normal':
        model = nn.DataParallel(c_resunet(arch='c-ResUnet', n_features_start=16, n_out=1, c0=c0,
                pretrained = False, progress= True)).to(device)
        try:
            model.load_state_dict(torch.load(model_path))
        except:
            model.load_state_dict(torch.load(model_path)['model_state_dict'])


    metrics_path = save_metrics_path + 'metrics/'
    summary_path = save_metrics_path + 'summary/'

    if os.path.exists(metrics_path):
        print('path already exist')
    else:
        os.makedirs(metrics_path)
    if os.path.exists(summary_path):
        print('path already exist')
    else:
        os.makedirs(summary_path)

    if not args.only_test:
        # dataloader
        if transform is None:
            transform = T.Compose([T.Lambda(lambda x: x * 1. / 255),
                           T.ToTensor()])
        cells_images = CellsLoader(val_images_path , val_masks_path ,
                               val_split=0.3, transform = transform, exclude_RT=True)
        data_loader = DataLoader(cells_images, batch_size=batch_size, shuffle=False)
        filenames = cells_images.imgs_list

        # model inference
        images, targets, preds = model_inference(data_loader, model, vae_flag)
        ths = np.linspace(th_min, th_max, steps)

        for th in ths:
            preds_processed, targets = post_processing(preds, targets, th, min_obj_size=min_obj_size,
                        foot=foot, area_threshold=area_threshold, max_dist=max_dist)
            metrics, summary = compute_metrics(targets, preds_processed, filenames)
            metrics.to_csv(metrics_path + 'threshold_{}.csv'.format(str(th)))
            summary.to_csv(summary_path + 'threshold_{}.csv'.format(str(th)))


    if args.test:
        summary_names = os.listdir(summary_path)
        summary_names_cleaned = []
        f1_scores = []
        ths = []
        precisions = []
        recalls = []
        for sn in summary_names:
            try:
                ths.append(round(float(sn.split('_')[1].strip('.csv')), 2))
                summary_names_cleaned.append(sn)
            except:
                print('exclunding ', sn)

        ixs = np.argsort(ths)
        ths.sort()
        summary_names_cleaned = [summary_names_cleaned[i] for i in ixs]

        for sn in summary_names_cleaned:
            summary = pd.read_csv(os.path.join(summary_path, sn))
            f1_scores.append(summary['F1_score'].values)
            precisions.append(summary['precision'].values)
            recalls.append(summary['recall'].values)

        f1_scores = [x[0] for x in f1_scores]
        kn = KneeLocator(ths, f1_scores, curve='concave', direction='decreasing')
        threshold_max_ix = np.argmax(f1_scores)
        threshold_max = ths[threshold_max_ix]

        threshold_knee = kn.knee
        threshold_knee = float(kn.knee) # df.F1.idxmax()

        if transform is None:
            transform = T.Compose([T.Lambda(lambda x: x * 1. / 255),
                                   T.ToTensor()])
        cells_images = CellsLoader(test_images_path, test_masks_path,
                                   val_split=0.3, transform=transform, exclude_RT=True)
        filenames = cells_images.imgs_list
        data_loader = DataLoader(cells_images, batch_size=batch_size, shuffle=False)

        images, targets, preds = model_inference(data_loader, model, vae_flag)

        if args.thresh_opt == 'kneedle':
            thresh = threshold_knee
            preds_processed, targets = post_processing(preds, targets, threshold_knee, min_obj_size=min_obj_size,
                                                       foot=foot, area_threshold=area_threshold, max_dist=max_dist)
        else:
            thresh = threshold_max
            preds_processed, targets = post_processing(preds, targets, threshold_max, min_obj_size=min_obj_size,
                                                       foot=foot, area_threshold=area_threshold, max_dist=max_dist)

        metrics, summary = compute_metrics(targets, preds_processed, filenames)

        metrics.to_csv(metrics_path + 'metrics_with_{}_threshold_{}.csv'.format(args.thresh_opt, str(thresh)))
        summary.to_csv(summary_path + 'summary_with_{}_threshold_{}.csv'.format(args.thresh_opt, str(thresh)))

        print(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Define parameters for test.')
    #parser.add_argument('--save_model_path', nargs="?", default=model_results,
    #                    help='the folder including the masks to crop')
    #parser.add_argument('--model_name', nargs="?", default='c-resunet',
    parser.add_argument('--test', action='store_const', const=True, default=True, help='make evaluation on test')
    parser.add_argument('--only_test', action='store_const', const=True, default=False, help='make evaluation on test')
    parser.add_argument('--architecture', default='normal', help='type of architecture to test [normal, vae]')
    parser.add_argument('--thresh_opt', nargs="?", default='max', help='[kneedle, max]')
    args = parser.parse_args()

    model_name = "c-resunet_g_aug_fix.h5" #"c-resunet_yellow_34_ft_green_4_unfr_dec.h5"
    #model_name = "c-resunet_y_11_dec_bottl_fs_20_rs_100.h5"

    model_train_list = ['supervised', 'weakly_supervised', 'fine_tuning', 'few_shot']
    model_train = model_train_list[0]
    #model_train = model_train_list[1:]
    if isinstance(model_train,list):
        model_train = '/'.join(model_train)
    model_path = '../model_results/{}/green/{}/{}'.format(model_train, model_name.replace('.h5', ''), model_name)
    save_metrics_path = '../model_results/{}/green/{}/'.format(model_train, model_name.replace('.h5',''))

    if 'few_shot' in model_train:
        if 'rs' in model_name:
            rs = int(model_name.replace('.h5', '').split('_')[-1])
            num_samples = int(model_name.replace('.h5', '').split('_')[-3])
            val_images_path = str(FewShot.as_posix().replace('/evaluation', '')) + '/images_{}_{}'.format(num_samples,rs)
            val_masks_path = str(FewShot.as_posix().replace('/evaluation', '')) + '/masks_{}_{}'.format(num_samples,rs)
            batch_size = 4
            print('val images', val_images_path)
            print('val masks', val_masks_path)
        else:
            num_samples = int(model_name.replace('.h5', '').split('_')[-1])
            val_images_path = str(FewShot.as_posix().replace('/evaluation', '')) + '/images_{}'.format(num_samples)
            val_masks_path = str(FewShot.as_posix().replace('/evaluation', '')) + '/masks_{}'.format(num_samples)
            batch_size = 4
    else:
        val_images_path = str(train_val_images.as_posix().replace('/evaluation', ''))
        val_masks_path = str(train_val_masks.as_posix().replace('/evaluation', ''))
        batch_size = 4

    test_images_path = str(test_images.as_posix().replace('/evaluation', ''))
    test_masks_path = str(test_masks.as_posix().replace('/evaluation', ''))

    print('analyzing model', model_name)

    metric_reports_th_sweep(args, Path(val_images_path), Path(val_masks_path),
                            Path(test_images_path), Path(test_masks_path), model_path, save_metrics_path=save_metrics_path,
                            batch_size=batch_size, c0=True, transform=None, th_min=0.3, th_max=0.90,
                            steps=10, min_obj_size=4, foot=4, area_threshold=6, max_dist=3)
                            # min_obj_size wa used with 2 for all the zperiments
