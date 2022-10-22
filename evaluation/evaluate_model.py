# TODO: check imports and function definitions to exclude the ones that are no longer relevant

import argparse
from config import IMG_HEIGHT, IMG_WIDTH, train_val_images, train_val_masks, test_images, test_masks, root
#IMPORT LIBRARIES
import sys

from skimage.segmentation import watershed
from math import hypot
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
from evaluation_utils import *
from kneed import KneeLocator
from skimage.segmentation import watershed
sys.path.append('..')
sys.path.append('../dataset_loader')
sys.path.append('../model')
from dataset_loader.image_loader import *
from model.resunet import *
from utils import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn
from pathlib import Path

def main(args):
# repo_path = Path("/home/luca/PycharmProjects/cell_counting_yellow")
    if args.mode == "eval":
        IMG_PATH = Path(train_val_images) #repo_path / "DATASET_fluo/train_val/full_size/all_images/images"
        MASKS_PATH = Path(train_val_masks)#repo_path / "DATASET_fluo/train_val/full_size/all_masks/masks"
    elif args.mode == "test":
        IMG_PATH = Path(test_images)#repo_path / "DATASET_fluo/test/all_images/all_images/images"
        MASKS_PATH = Path(test_masks)#repo_path / "DATASET_fluo/test/all_masks/all_masks/masks"
    else:
        IMG_PATH = Path(train_val_images)  # repo_path / "DATASET_fluo/train_val/full_size/all_images/images"
        MASKS_PATH = Path(train_val_masks)  # repo_path / "DATASET_fluo/train_val/full_size/all_masks/masks"


    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    c0 = True


    model = nn.DataParallel(c_resunet(arch='c-ResUnet', n_features_start=16, n_out=1, c0=c0,
                                      pretrained=False, progress=True)).to(device)
    model.load_state_dict(torch.load('../model_results/supervised/green/{}'.format(args.model_name)))

    save_path = '../model_results/supervised/green/{}'.format(args.mode)
    repo_path = Path(str(root).replace(root.split('/')[-1], ''))
    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_path.chmod(16886)  # chmod 776
    text = "\nReading images from: {}".format(str(IMG_PATH))
    print("#" * len(text))
    print(text)
    print("Output folder set to: {}\n".format(str(save_path)))

    print("#" * len(text))
    print(f"\nModel: {args.model_name}\n\n")


    # predict with generator
    transform = T.Compose([T.Lambda(lambda x: x * 1. / 255),
                           T.ToTensor()
                           ])

    cells_images = CellsLoader(repo_path + '/data/test/original_images/',
                                repo_path + '/data/test/original_masks/',
                               val_split=0.3, transform = transform)

    filenames = os.listdir(IMG_PATH)
    batch_size = 4
    test_loader = DataLoader(cells_images, batch_size=batch_size)

    model.eval()
    predict = []
    masks = []
    images = []
    for ix, (x, y) in enumerate(test_loader):
        print("batch {}".format(ix))
        with torch.no_grad():
            results = model(x.to(device)).cpu().detach()
            predict.extend(results)
            torch.cuda.empty_cache()
            images.extend(x)
            masks.extend(y)

     ########################################
    nb_samples = len(filenames)

    opt_thresh_path = repo_path / "results/eval" / 'metrics_{}.csv'.format(args.model_name)
    df = pd.read_csv(opt_thresh_path, index_col='Threshold')
    if args.threshold == 'best':
        threshold_seq = [df.F1.idxmax()]
    elif args.threshold == 'knee':
        x = df.index
        y = df.F1
        kn = KneeLocator(x, y, curve='concave', direction='decreasing')
        threshold_seq = [kn.knee]  # df.F1.idxmax()
    else:
        threshold_seq = np.arange(start=0.5, stop=0.98, step=0.025)

    metrics_df_validation_rgb = pd.DataFrame(None, columns=["F1", "MAE", "MedAE", "MPE", "accuracy",
                                                            "precision", "recall"])

    for _, threshold in tqdm(enumerate(threshold_seq), total=len(threshold_seq)):

        print(f"Running for threshold: {threshold:.3f}")
        # create dataframes for storing performance measures
        validation_metrics_rgb = pd.DataFrame(
            columns=["TP", "FP", "FN", "Target_count", "Predicted_count"])
        # loop on masks
        for idx, img_path in enumerate(filenames):
            mask_path = MASKS_PATH / img_path.split("/")[1]
            pred_mask_rgb = predict_mask_from_map(predict[idx], threshold)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            compute_metrics(pred_mask_rgb, mask,
                            validation_metrics_rgb, img_path.split("/")[1])
        metrics_df_validation_rgb.loc[threshold] = F1Score(validation_metrics_rgb)
    outname = save_path / 'metrics_{}.csv'.format(model_name[:-3])
    metrics_df_validation_rgb.to_csv(outname, index=True, index_label='Threshold')
    _ = plot_thresh_opt(metrics_df_validation_rgb, model_name, save_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run evaluation pipeline for specified model name')
    parser.add_argument('model_name', metavar='name', type=str, default="ResUnet",  # nargs='+',
                        help='Name of the model to evaluate.')
    parser.add_argument('--out_folder', metavar='folder', type=str, default="results",
                        help='Output folder')
    parser.add_argument('--batch_size', metavar='batch_size', type=int, default=2,
                        help='Batch size for generator used for predictions')
    parser.add_argument('--mode', metavar='mode', type=str, default="eval",
                        help="""Running mode. Valid values:
                                - eval (default) --> optimise threshold (train_val folder, full size images)                            
                                - test --> validate on test images (test folder, full size images)                     
                                - test_code --> for testing changes in the code
                                """)
    parser.add_argument('--threshold', metavar='threshold', type=str, default='grid',
                        help='Whether to use a threshold optimized on the validation set (`best` for argmax or `knee` for kneedle) or grid of values')
    args = parser.parse_args()

    main(args)



