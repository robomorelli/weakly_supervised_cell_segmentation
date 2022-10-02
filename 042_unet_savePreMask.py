#IMPORT LIBRARIES
import sys
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Conv2D, UpSampling2D, Lambda
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K
from keras import initializers, layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
from keras import optimizers
import keras.applications as ka
from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.utils import set_trainable
from subprocess import check_output
from keras.optimizers import Adam

from tqdm import tqdm
import cv2
from skimage.transform import resize

from config_script import *


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def savePreMask():
    
    test_ids = check_output(["ls", ALL_IMAGES]).decode("utf8").split()
    model = load_model(str(MODEL_CHECKPOINTS / 'UnetRGB_crop_v3.h5'), custom_objects={'mean_iou': mean_iou, 'dice_coef': dice_coef})
    
    for idx, name in tqdm(enumerate(test_ids),total=len(test_ids)):
        
        img_x = cv2.imread(str(ALL_IMAGES / name))
        #img_x = resize(img_x, (400, 400), mode='constant', preserve_range=True)
        img_x = img_x*1./255
        img_x = np.expand_dims(img_x, axis=0)
        img = model.predict(img_x,verbose = 1)
        print(img.shape)
        cv2.imwrite(str(PRED_IMAGES/name),np.squeeze(img))    

savePreMask()