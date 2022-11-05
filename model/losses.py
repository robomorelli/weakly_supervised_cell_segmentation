from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import math
import sys
from tqdm import tqdm

sys.path.append('..')
from config import *
from dataset_loader.image_loader import *


def WeightedLoss(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_pred, y_true):

        #b_ce = nn.BCEWithLogitsLoss(reduction = 'none')(y_true[:,0:1,:,:].float(), y_pred[:,0:1,:,:].float()) #try without reduction
        b_ce = nn.BCELoss(reduction='none')(y_pred[:, 0:1, :, :].float(), y_true[:, 0:1, :, :].float())
        # Apply the weights
        class_weight_vector = y_true[:,0:1,:,:] * one_weight + (1. - y_true[:,0:1,:,:]) * zero_weight

        weight_vector = class_weight_vector * y_true[:,1:2,:,:]
        weighted_b_ce = weight_vector * b_ce

        #we should first make a sum reduction on rows and column and then take a mean over element of batch
        #s1 = torch.sum(weighted_b_ce, axis=-2)
        #s2 = torch.sum(s1, axis=-1)

        # Return the mean error
        return torch.mean(weighted_b_ce)

    return weighted_binary_crossentropy

def WeightedLossAE(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_pred, y_true):

        #channel = np.random.randint(0,2,1)
        b_ce = nn.BCELoss(reduction='none')(y_pred[:, 0:1, :, :].float(), y_true[:, 0:1, :, :].float())
        # Apply the weights
        weighted_b_ce = y_true[:,1:2,:,:] * b_ce

        #we should first make a sum reduction on rows and column and then take a mean over element of batch
        #s1 = torch.sum(weighted_b_ce, axis=-2)
        #s2 = torch.sum(s1, axis=-1)

        # Return the mean error
        return torch.mean(weighted_b_ce)

    return weighted_binary_crossentropy

def UnweightedLoss(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_pred, y_true):

        #b_ce = nn.BCEWithLogitsLoss(reduction = 'none')(y_true[:,0:1,:,:].float(), y_pred[:,0:1,:,:].float()) #try without reduction
        b_ce = nn.BCELoss(reduction='none')(y_pred[:, 0:1, :, :].float(), y_true[:, 0:1, :, :].float())
        # Apply the weights
        class_weight_vector = y_true[:,0:1,:,:] * one_weight + (1. - y_true[:,0:1,:,:]) * zero_weight
        weighted_b_ce = class_weight_vector * b_ce

        # Return the mean error
        return torch.mean(weighted_b_ce)

    return weighted_binary_crossentropy

def KL_loss_forVAE(mu, sigma):
    mu_prior = torch.tensor(0)
    sigma_prior = torch.tensor(1)
    kl_loss = torch.mul(torch.mul(sigma, sigma), torch.mul(sigma_prior,sigma_prior))
    div = torch.div(mu_prior - mu, sigma_prior)
    kl_loss += torch.mul(div, div)
    kl_loss += torch.log(torch.div(sigma_prior, sigma)) -1
    #kl_loss = torch.sum(kl_loss, axis=(-1,-2))
    ##return 0.5 * torch.sum(kl_loss)
    return 0.5 * torch.sum(kl_loss)


def loss_VAE(mu, sigma, x):

    au = 0.5*torch.log(2*np.pi*(sigma*sigma)) #aleatoric uncertainty
    ne = (torch.square(mu - x))/(2*torch.square(sigma))#normalized error
    nll_loss = au + ne
    sigma_mean = torch.mean(sigma, axis=1)
    #return torch.mean(torch.sum(nll_loss, dim=[-1.-2])), au, ne
    return torch.mean(nll_loss), torch.mean(au, axis=1), torch.mean(ne, axis=1), sigma_mean

def WeightedNLLLoss(zero_weight, one_weight):

    def weighted_NLL(mu, sigma, y):

        au = 0.5*torch.log(2*np.pi*(sigma*sigma)) #aleatoric uncertainty
        ne = (torch.square(mu - y))/(2*torch.square(sigma))#normalized error
        nll_loss = au + ne

        # Apply the weights
        class_weight_vector = y[:,0:1,:,:] * one_weight + (1. - y[:,0:1,:,:]) * zero_weight

        weight_vector = class_weight_vector * y[:,1:2,:,:]
        weighted_nll = weight_vector * nll_loss

        # Return the mean error
        return torch.mean(weighted_nll)
    return weighted_NLL

def loss_VAE_rec(mu, sigma, x):

    au = 0.5*torch.log(2*np.pi*(sigma*sigma)) #aleatoric uncertainty
    ne = (torch.square(mu - x))/(2*torch.square(sigma))#normalized error
    nll_loss = au + ne
    ## sum over channel to get the total sigma for each pixel
    au_squared = torch.square(au)
    au_1ch = torch.sqrt(torch.sum(au_squared, axis=1))
    ne_squared = torch.square(ne)
    ne_1ch = torch.sqrt(torch.sum(ne_squared, axis=1))

    #return torch.sum(nll_loss), au, ne
    return torch.mean(nll_loss), au, ne, au_1ch, ne_1ch