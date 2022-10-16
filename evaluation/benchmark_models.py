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
matplotlib.use('qt5Agg')

from evaluation.evaluation_utils import post_processing, compute_metrics_global, model_inference, compute_metrics

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

def metric_reports_th_sweep(images_path, targets_path, model_path, save_metrics_path,
                            batch_size=4, c0=True, transform=None, th_min=0.3, th_max=1,
                            steps=5, min_obj_size=2, foot=4, area_threshold=6, max_dist=3):
    # load model
    model = nn.DataParallel(c_resunet(arch='c-ResUnet', n_features_start=16, n_out=1, c0=c0,
            pretrained = False, progress= True)).to(device)
    model.load_state_dict(torch.load(model_path))
    model_name = model_path.split('/')[-1]

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

    # dataloader
    if transform is None:
        transform = T.Compose([T.Lambda(lambda x: x * 1. / 255),
                       T.ToTensor()])
    cells_images = CellsLoader(images_path, targets_path,
                           val_split=0.3, transform = transform, exclude_RT=True)
    data_loader = DataLoader(cells_images, batch_size=batch_size, shuffle=False)
    filenames = cells_images.imgs_list

    # model inference
    images, targets, preds = model_inference(data_loader, model)
    ths = np.linspace(th_min, th_max, steps)

    for th in ths:
        preds_processed, targets = post_processing(preds, targets, th, min_obj_size=min_obj_size,
                    foot=foot, area_threshold=area_threshold, max_dist=max_dist)
        metrics, summary = compute_metrics(targets, preds_processed, filenames)
        metrics.to_csv(metrics_path + 'threshold_{}.csv'.format(str(th)))
        summary.to_csv(summary_path + 'threshold_{}.csv'.format(str(th)))

if __name__ == "__main__":
    model_name = "c-resunet_21.h5"
    test_images_path = str(test_images.as_posix().replace('/evaluation', ''))
    test_masks_path = str(test_masks.as_posix().replace('/evaluation', ''))
    model_path = '../model_results/supervised/green/{}'.format(model_name)
    save_metrics_path = '../model_results/supervised/green/{}/'.format(model_name.replace('.h5',''))

    metric_reports_th_sweep(Path(test_images_path), Path(test_masks_path), model_path, save_metrics_path=save_metrics_path,
                            batch_size=4, c0=True, transform=None, th_min=0.3, th_max=1,
                            steps=10, min_obj_size=2, foot=4, area_threshold=6, max_dist=3)
