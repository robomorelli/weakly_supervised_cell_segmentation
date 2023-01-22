import random as rn
import numpy as np
from model.resunet import *
from model.utils import EarlyStopping, WeightedLoss, load_data_train_eval, UnweightedLoss, WeightedLossAE
from dataset_loader.image_loader import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms as T
import torch.nn as nn
from pathlib import Path
import torch.multiprocessing as mp
from tqdm import tqdm
from time import sleep
import torch
import sys
import argparse
from pytorch_metric_learning import losses, reducers, miners, distances, regularizers
sys.path.append('..')
from config import *
from model.utils import train_cycle, vae_train_cycle

np.random.seed(40)
# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
#rn.seed(33323)
rn.seed(3334)

def train(args):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print('added visible gpu')
        ndevices = torch.cuda.device_count()

    else:
        ndevices = 1

    num_workers = 0
    if args.ae or args.ae_bin:
        n_out = 3
        args.ae = True
    else:
        n_out = 1

    train_loader, val_loader = load_data_train_eval(dataset=args.dataset,
                                                           batch_size=args.bs, validation_split=0.3,
                                                           grayscale=False, num_workers=num_workers,
                                                           shuffle_dataset=True, random_seed=42, ngpus=ndevices,
                                                           ae=args.ae,
                                                           few_shot=args.few_shot, num_samples=args.few_shot_samples,
                                                           few_shot_random_seed=args.few_shot_random_seed,
                                                           weakly_supervised=args.weakly_supervised,
                                                           massive=args.massive)

    # Take before the resume path and then change model name otherwise we don't found the model_name  to resume (because it change to new_model_name)
    if args.resume or args.fine_tuning:
        resume_path = str(args.resume_path) + '/{}/{}'.format(args.model_name, args.model_name + '.h5')
        if args.few_shot_random_seed:
            args.model_name = args.new_model_name + '_rs_{}'.format(args.few_shot_random_seed)
            print(args.model_name)
        else:
            args.model_name = args.new_model_name
    elif args.few_shot:
        if args.few_shot_random_seed:
            args.model_name = args.model_name + '_rs_{}'.format(args.few_shot_random_seed)

    if args.fine_tuning:
        if args.few_shot:
            added_path = '/fine_tuning/few_shot/{}/{}/'.format(args.dataset, args.model_name)
        else:
            added_path = '/fine_tuning/{}/{}/'.format(args.dataset, args.model_name)
    elif args.ae:
        added_path = '/autoencoder/{}/{}/'.format(args.dataset, args.model_name)
    elif args.ae_bin:
        added_path = 'autoencoder_bin/{}/{}/'.format(args.dataset, args.model_name)
    elif args.few_shot:
        added_path = '/few_shot/{}/{}/'.format(args.dataset, args.model_name)
    elif args.weakly_supervised:
        added_path = '/weakly_supervised/{}/{}/'.format(args.dataset, args.model_name)
    else:
        if args.massive:
            added_path = '/supervised_massive/{}/{}/'.format(args.dataset, args.model_name.replace('h5', ''), args.model_name)
        else:
            added_path = '/supervised/{}/{}/'.format(args.dataset, args.model_name.replace('h5', ''),
                                                                args.model_name)

    if os.path.exists(str(args.save_model_path) + added_path):
        print("path exists")
    else:
        os.makedirs(str(args.save_model_path) + added_path)

    if args.resume or args.fine_tuning:
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            if device == 'cpu':
                checkpoint = torch.load(resume_path)
            else:
                if args.from_ae:
                    # to replace n_out = 3 with ###new weight#### for a binary layer on top (use to fine tune an autoencoder)
                    model = load_ae_to_bin(resume_path=resume_path, device=device, n_features_start=16, n_out=n_out,
                                           fine_tuning=args.fine_tuning, unfreezed_layers=args.unfreezed_layers)
                else:
                    if args.vae:
                        model = load_model(resume_path=resume_path, device=device, n_features_start=16, n_out=n_out,
                                           fine_tuning=args.fine_tuning, unfreezed_layers=args.unfreezed_layers, vae =True
                                           ,ae_bin=args.ae_bin)
                    else:
                        model = load_model(resume_path=resume_path, device=device, n_features_start=16, n_out=n_out,
                                           fine_tuning=args.fine_tuning, unfreezed_layers=args.unfreezed_layers, ae_bin=args.ae_bin)
            #if args.new_model_name:
            #    args.model_name = args.new_model_name

        # No checkpoint, model form scratch with or without co
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))
            if args.c0:
                model = nn.DataParallel(c_resunet(arch='c-ResUnet', n_features_start=16, n_out=n_out,
                                                  pretrained=False, progress=True, ae_bin=args.ae_bin,)).to(device)
            else:
                model = nn.DataParallel(c_resunet(arch='c-ResUnet', n_features_start=16, n_out=n_out, c0=False,
                                                  pretrained=False, progress=True, ae_bin=args.ae_bin,)).to(device)
    else:
        if args.c0:
            model = nn.DataParallel(c_resunet(arch='c-ResUnet', n_features_start=16, n_out=n_out,
                                              pretrained=False, progress=True, ae_bin=args.ae_bin,
                                              device=device).to(device))
        else:
            model = nn.DataParallel(
                c_resunet(arch='c-ResUnet', n_features_start=16, n_out=n_out, c0=False,
                          pretrained=False, progress=True, ae_bin=args.ae_bin,
                          device=device).to(device))

    val_loss = 10 ** 16
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=args.patience_lr,
                                                           threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                           min_lr=9e-8, verbose=True)
    early_stopping = EarlyStopping(patience=args.patience)
    # Train Loop####
    """
    Set the model to the training mode first and train
    """

    if args.loss_type == 'weighted':
        criterion = WeightedLoss(1, 1.5) #also weighted maps for touching cells
    elif args.loss_type == 'unweightedAE':
        criterion = nn.BCELoss() #bce for three channel input and target
    elif args.loss_type == 'unweighted':
        criterion = UnweightedLoss(1, 1.5)  # take only the first channel of the mask and weight with class
    # torch.autograd.set_detect_anomaly(True)

    if args.vae:
        vae_train_cycle(model, criterion, optimizer, train_loader, val_loader, device,
                    args.save_model_path, added_path, scheduler, early_stopping, args.model_name, epochs=100)
    else:
        train_cycle(model, criterion, optimizer, train_loader, val_loader, device,
                    args.save_model_path, added_path, scheduler, early_stopping, args.model_name, epochs=100)



if __name__ == "__main__":
    ###############################################
    # TO DO: add parser for parse command line args
    ###############################################
    parser = argparse.ArgumentParser(description='Define parameters for test.')
    parser.add_argument('--save_model_path', nargs="?", default=model_results,
                        help='the folder including the masks to crop')
    parser.add_argument('--model_name', nargs="?", default='c-resunet',
                        help='model_name')
    parser.add_argument('--new_model_name', nargs="?", default='c-resunet',
                        help='the name the model will have after resume another model name')
    parser.add_argument('--loss_type', nargs="?", default='unweighted',
                        help='what kind of loss among weighted, unweightedAE')
    parser.add_argument('--patience', nargs="?", type=int, default=10, help='patience for early stopping')
    parser.add_argument('--patience_lr', nargs="?", type=int, default=3, help='patience for learning rate')
    parser.add_argument('--lr', nargs="?", type=float, default= 0.001, help='learning rate value')
    parser.add_argument('--epochs', nargs="?", type=int, default=200, help='number of epochs')
    parser.add_argument('--bs', nargs="?", type=int, default=8, help='batch size')
    parser.add_argument('--dataset', nargs="?", default='green', help='dataset flavour')

    parser.add_argument('--c0', type=int,default=1,
                        help='include or not c0 lauyer')
    parser.add_argument('--ae', action='store_true',
                        help='autoencoder train of resunet')
    parser.add_argument('--ae_bin', action='store_true',
                        help='autoencoder train of resunet')
    parser.add_argument('--ae_no_c0', action='store_true',
                        help='autoencoder without c0')
    parser.add_argument('--few_shot', action='store_true',
                        help='use a small dataset to train the model')
    parser.add_argument('--few_shot_random_seed', default=10,
                        help='random seed associated with few shot data generation')
    parser.add_argument('--few_shot_samples', type=int, default=100,
                        help='how many images')
    parser.add_argument('--massive', action='store_true',
                        help='how many images')
    parser.add_argument('--weakly_supervised', action='store_true',
                        help='use a small dataset to train the model')
    parser.add_argument('--resume', action='store_true',
                        help='resume training of the model specified with the model name')
    parser.add_argument('--resume_path', default=model_results_supervised_yellow,
                        help='checkpoint to load to train or fine tune')
    parser.add_argument('--fine_tuning', action='store_true',
                        help='fine tune the model or not')
    parser.add_argument('--vae', action='store_const', const=True, default=False, help='make evaluation on test')
    parser.add_argument('--from_ae', action='store_true',
                        help='fine tune the model coming from autoencoder with pre binary layer')
    parser.add_argument('--unfreezed_layers', nargs='+', default=1,
                        help='number of layer to unfreeze for fine tuning can be a number or a block [encoder, decoder, head]'
                             'if only one number it start from last layer and unfreeze n layers'
                             'if 2 numbers and the first one is zero, it start from the beginning of the model and unfreeze n layers'
                             'if 2 number but the firt one is not a zero it will be the first layer unfreezed untill the second number layer is reached'
                             'if need to unfrezze couple by couple the conv block to unfreeze:'
                             '1 1 colorspace conv_block bottleneck 1 :first residual_block and first upconv (1, 1) are unfreezed togheter '
                             'with colorspace (colorspace) first conv block (conv_block)'
                             'and last layer (the last 1 in the series)')
    args = parser.parse_args()

    if not (Path(args.save_model_path).exists()):
        print('creating path')
        os.makedirs(args.save_model_path)
    train(args=args)

#next to do:
#--few_shot --fine_tuning --lr 0.0001 --model_name c-resunet_y_11 --unfreezed_layers 5 --new_model_name c-resunet_y_11_dec_bottl_fs_200  --few_shot_samples 200 --few_shot_random_seed 123
#--few_shot --fine_tuning --model_name c-resunet_y_11 --unfreezed_layers 5 --new_model_name c-resunet_y_11_dec_bottl_fs_40  --few_shot_samples 40 (new training modalities)
#--few_shot --fine_tuning --model_name c-resunet_y_11 --unfreezed_layers 1 --new_model_name c-resunet_y_11_last_fs_40  --few_shot_samples 40

#--few_shot --lr 0.001 --model_name c-resunet_g_fs_30 --few_shot_samples 30

#Template
# --fine_tuning  --unfreezed_layers 5 --model_name c-resunet_y_3 --new_model_name c-resunet_y_3_dec_bottl