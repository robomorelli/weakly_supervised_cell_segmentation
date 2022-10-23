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
    if args.ae:
        n_out = 3
        args.ae = True
    else:
        n_out = 1

    train_loader, validation_loader = load_data_train_eval(dataset=args.dataset,
                                                           batch_size=args.bs, validation_split=0.3,
                                                           grayscale=False, num_workers=num_workers,
                                                           shuffle_dataset=True, random_seed=42, ngpus=ndevices,
                                                           ae=args.ae,
                                                           few_shot=args.few_shot)
    if args.resume or args.fine_tuning:
        resume_path = str(args.resume_path) + '/{}/{}'.format(args.model_name, args.model_name + '.h5')
        args.model_name = args.new_model_name

    if args.fine_tuning:
        if args.few_shot:
            added_path = '/fine_tuning/few_shot/{}/{}/'.format(args.dataset, args.model_name)
        else:
            added_path = '/fine_tuning/{}/{}/'.format(args.dataset, args.model_name)
    elif args.ae:
        added_path = '/autoencoder/{}/{}/'.format(args.dataset, args.model_name)
    elif args.few_shot:
        added_path = '/few_shot/{}/{}/'.format(args.dataset, args.model_name)
    else:
        added_path = '/supervised/{}/{}/{}/'.format(args.dataset, args.model_name.replace('h5', ''), args.model_name)

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
                    model = load_model(resume_path=resume_path, device=device, n_features_start=16, n_out=n_out,

                                       fine_tuning=args.fine_tuning, unfreezed_layers=args.unfreezed_layers)
            if args.new_model_name:
                args.model_name = args.new_model_name

        # No checkpoint, model form scratch with or without co
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))
            if args.c0:
                model = nn.DataParallel(c_resunet(arch='c-ResUnet', n_features_start=16, n_out=n_out,
                                                  pretrained=False, progress=True)).to(device)
            else:
                model = nn.DataParallel(c_resunet(arch='c-ResUnet', n_features_start=16, n_out=n_out, c0=False,
                                                  pretrained=False, progress=True)).to(device)
    else:
        if args.c0:
            model = nn.DataParallel(c_resunet(arch='c-ResUnet', n_features_start=16, n_out=n_out,
                                              pretrained=False, progress=True,
                                              device=device).to(device))
        else:
            model = nn.DataParallel(
                c_resunet(arch='c-ResUnet', n_features_start=16, n_out=n_out, c0=False,
                          pretrained=False, progress=True,
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
        criterion = WeightedLoss(1, 1.5)
    if args.loss_type == 'weightedAE':
        criterion = WeightedLossAE(1, 1)
    if args.loss_type == 'unweightedAE':
        criterion = nn.BCELoss()
    elif args.loss_type == 'unweighted':
        criterion = UnweightedLoss(1, 1.5)  # not for autoencoder but segmentation weighted only on the class type
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.epochs):
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, (image, target) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()

                y = target.to(device)
                x = image.to(device)
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
            ###############################################
            # eval mode for evaluation on validation dataset_loader
            ###############################################
            with torch.no_grad():
                model.eval()
                temp_val_loss = 0
                with tqdm(validation_loader, unit="batch") as vepoch:
                    for i, (image, target) in enumerate(vepoch):
                        optimizer.zero_grad()

                        y = target.to(device)
                        x = image.to(device)
                        out = model(x)
                        loss = criterion(out, y)
                        temp_val_loss += loss
                        if i % 10 == 0:
                            print("VALIDATION Loss: {} batch {} on total of {}".format(loss.item(),
                                                                                       i, len(validation_loader)))

                    temp_val_loss = temp_val_loss / len(validation_loader)
                    print('validation_loss {}'.format(temp_val_loss))
                    scheduler.step(temp_val_loss)
                    if temp_val_loss < val_loss:
                        print('val_loss improved from {} to {}, saving model to {}' \
                          .format(val_loss, temp_val_loss, args.save_model_path.as_posix() + '/' + added_path + args.model_name))
                        print("saving model to {}".format(args.save_model_path.as_posix() + '/' + added_path + args.model_name))
                        path_posix = args.save_model_path.as_posix() + '/' + added_path + args.model_name
                        save_path = path_posix + '.h5'
                        torch.save(model.state_dict(), save_path)
                        val_loss = temp_val_loss

                    early_stopping(temp_val_loss)
                    if early_stopping.early_stop:
                        break

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
    parser.add_argument('--patience', nargs="?", type=int, default=5, help='patience for checkpoint')
    parser.add_argument('--patience_lr', nargs="?", type=int, default=3, help='patience for early stopping')
    parser.add_argument('--lr', nargs="?", type=int, default= 0.0001, help='learning rate value')
    parser.add_argument('--epochs', nargs="?", type=int, default=200, help='number of epochs')
    parser.add_argument('--bs', nargs="?", type=int, default=8, help='batch size')
    parser.add_argument('--dataset', nargs="?", default='green', help='dataset flavour')

    parser.add_argument('--c0', type=int,default=1,
                        help='include or not c0 lauyer')
    parser.add_argument('--ae', action='store_true',
                        help='autoencoder train of resunet')
    parser.add_argument('--ae_no_c0', action='store_true',
                        help='autoencoder without c0')
    parser.add_argument('--few_shot', action='store_true',
                        help='use a small dataset to train the model')
    parser.add_argument('--resume', action='store_true',
                        help='resume training of the model specified with the model name')
    parser.add_argument('--resume_path', default=model_results_supervised_yellow,
                        help='checkpoint to load to train or fine tune')
    parser.add_argument('--fine_tuning', action='store_true',
                        help='fine tune the model or not')
    parser.add_argument('--from_ae', action='store_true',
                        help='fine tune the model coming from autoencoder with pre binary layer')
    parser.add_argument('--unfreezed_layers', default=1,
                        help='number of layer to unfreeze for fine tuning can be a number or a block [encoder, decoder, head]')
    args = parser.parse_args()

    if not (Path(args.save_model_path).exists()):
        print('creating path')
        os.makedirs(args.save_model_path)
    train(args=args)

