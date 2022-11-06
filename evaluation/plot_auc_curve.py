import os.path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import argparse
sys.path.append('..')
from kneed import KneeLocator
from evaluation.evaluation_utils import post_processing, compute_metrics, model_inference
from dataset_loader.image_loader import CellsLoader
from config import *

import matplotlib
matplotlib.use('QtAgg')

def main(summary_path):

    if args.all:
        models_names = os.listdir(summary_path)
        for mod_name in models_names:
            folders = os.listdir(os.path.join(summary_path, mod_name))
            summary_names_cleaned = []
            f1_scores = []
            ths = []
            precisions = []
            recalls = []
            if "summary" in folders:
                summary_names = os.listdir(os.path.join(summary_path, mod_name) + '/summary')
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
                    summary = pd.read_csv(os.path.join(summary_path, mod_name)  + '/summary/' + sn)
                    f1_scores.append(summary['F1_score'].values)
                    precisions.append(summary['precision'].values)
                    recalls.append(summary['recall'].values)

                f1_scores = [x[0] for x in f1_scores]
                plt.plot(ths, f1_scores)
                plt.title('{}'.format(mod_name))
                plt.savefig(os.path.join(summary_path, mod_name) + '/f1_plot_train_val.png')
                plt.close()
            else:
                continue

    else:
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

        plt.plot(ths, f1_scores)
        plt.savefig(os.path.join(summary_path, mod_name) + 'f1_plot_train_val.png')


if __name__ == "__main__":
    ###############################################
    # TO DO: add parser for parse command line args
    ###############################################
    parser = argparse.ArgumentParser(description='Define parameters for test.')
    parser.add_argument('--model_path', nargs="?", default=model_results,
                        help='the folder including the masks to crop')
    parser.add_argument('--training_type', nargs="?", default='fine_tuning',
                        help='[fine_tuning, supervised, few_shot]')
    parser.add_argument('--model_name', nargs="?", default='c-resunet_21', help='model_name')
    parser.add_argument('--dataset', nargs="?", default='green', help='[green, yelow]')
    parser.add_argument('--all', action='store_const', const=True, default=True, help='make evaluation on test')
    args = parser.parse_args()

    if args.all:
        summary_path = '../model_results/{}/{}/'.format(args.training_type, args.dataset)
    else:
        summary_path = '../model_results/{}/{}/{}/'.format(args.training_type, args.dataset, args.model_name.replace('.h5','')) + 'summary/'

    main(summary_path)

