#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
import keras
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Add, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization,Conv2D, UpSampling2D, Lambda, ZeroPadding2D
from keras.models import Model, load_model
from keras.layers import Input, SpatialDropout2D, SeparableConv2D
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras import initializers, layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
from keras import optimizers
import keras.applications as ka
from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.utils import set_trainable

from importlib import import_module
plots = import_module('051_plotscalars')
tb = import_module('050_tensorboard')
lrate = import_module('043_learning_rate')

from config_script import *

tot_img_after_aug = count_files_in_directory(0, dir_list = ALL_IMAGES)

BATCH_SIZE = 2
VALID_BATCH_SIZE = 2
seed1=111 
seed2=222
seed3=333
val_percentage = 0.20
valid_number = int(tot_img_after_aug*val_percentage)
train_number = tot_img_after_aug - valid_number

IMG_CHANNELS = 3
IMG_HEIGHT = 512
IMG_WIDTH = 512

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred[:,:,:,0:1]> t)
        score, up_opt = tf.metrics.mean_iou(y_true[:,:,:,0:1], y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true[:,:,:,0:1])
    y_pred_f = K.flatten(y_pred[:,:,:,0:1])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def create_weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true[:,:,:,0:1], y_pred[:,:,:,0:1])

        # Apply the weights
        weight_vector = y_true[:,:,:,0:1]*one_weight + (1. - y_true[:,:,:,0:1])*zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy
    
def create_weighted_binary_crossentropy_2(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):

        b_ce = K.binary_crossentropy(y_true[:,:,:,0:1], y_pred[:,:,:,0:1])

        # Apply the weights
        weight_vector0 = y_true[:,:,:,0:1] * one_weight + (1. - y_true[:,:,:,0:1]) * zero_weight

        weight_vector = weight_vector0 * y_true[:,:,:,1:2]
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy
    

def join_generators(generators, n_models):
    while True: # keras requires all generators to be infinite
        data = [next(g) for g in generators]

        x = data[0]
        y = [data[1] for d in range(n_models)]

        yield x, y    
        
def disjoin_generators(generators):
    while True: # keras requires all generators to be infinite
        data = [next(g) for g in generators]

        x = [data[0], data[2], data[4]]
        y = [data[1],data[3], data[5]]
        
        if x[0].shape==x[1].shape==x[2].shape:

           yield x, y
        
def padder(x):
    
    x = tf.concat((x, tf.expand_dims(x[:, -1, :,:], 1)), axis=1)
    
    return(x)


def imageGenerator(color_mode = 'grayscale', n_outputs = 3):

    image_datagen = ImageDataGenerator(rescale=1./255,validation_split = val_percentage)
                                       #preprocessing_function=rgb2hsv)
    mask_datagen = ImageDataGenerator(rescale=1./255, validation_split = val_percentage, dtype=bool)
    
    train_image_generator = image_datagen.flow_from_directory(ALL_IMAGES.parent,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=BATCH_SIZE,
                                                              class_mode = None, seed = seed1, subset = 'training')
    train_mask_generator = mask_datagen.flow_from_directory(ALL_MASKS.parent,
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=BATCH_SIZE,
                                                            class_mode = None, seed = seed1, subset = 'training',
                                                            color_mode = color_mode)
    
    valid_image_generator = image_datagen.flow_from_directory(ALL_IMAGES.parent,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=VALID_BATCH_SIZE,
                                                              class_mode = None, seed = seed1, subset = 'validation')
    valid_mask_generator = mask_datagen.flow_from_directory(ALL_MASKS.parent,
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=VALID_BATCH_SIZE,
                                                            class_mode = None, seed = seed1, subset = 'validation',
                                                            color_mode = color_mode)
                                                            
    train_generator = join_generators((train_image_generator, train_mask_generator), n_outputs)
    valid_generator = join_generators((valid_image_generator, valid_mask_generator), n_outputs)                                                    
                                                            
    return train_generator, valid_generator     

def imageDisjointGenerator(color_mode = 'grayscale'):

    image_datagen = ImageDataGenerator(rescale=1./255,validation_split = val_percentage)
                                       #preprocessing_function=rgb2hsv)
    mask_datagen = ImageDataGenerator(rescale=1./255, validation_split = val_percentage, dtype=bool)
    
    train_image_generator = image_datagen.flow_from_directory(ALL_IMAGES.parent,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=BATCH_SIZE,
                                                              class_mode = None, seed = seed1, subset = 'training')
    train_mask_generator = mask_datagen.flow_from_directory(ALL_MASKS.parent,
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=BATCH_SIZE,
                                                            class_mode = None, seed = seed1, subset = 'training',
                                                            color_mode = color_mode)
                                                            
    train_image_generator_1 = image_datagen.flow_from_directory(ALL_IMAGES.parent,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=BATCH_SIZE,
                                                              class_mode = None, seed = seed2, subset = 'training')
    train_mask_generator_1 = mask_datagen.flow_from_directory(ALL_MASKS.parent,
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=BATCH_SIZE,
                                                            class_mode = None, seed = seed2, subset = 'training',
                                                            color_mode = color_mode)
                                                            
    train_image_generator_2 = image_datagen.flow_from_directory(ALL_IMAGES.parent,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=BATCH_SIZE,
                                                              class_mode = None, seed = seed3, subset = 'training')
    train_mask_generator_2 = mask_datagen.flow_from_directory(ALL_MASKS.parent,
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=BATCH_SIZE,
                                                            class_mode = None, seed = seed3, subset = 'training',
                                                            color_mode = color_mode)
    
    
    valid_image_generator = image_datagen.flow_from_directory(ALL_IMAGES.parent,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=VALID_BATCH_SIZE,
                                                              class_mode = None, seed = seed1, subset = 'validation')
    valid_mask_generator = mask_datagen.flow_from_directory(ALL_MASKS.parent,
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=VALID_BATCH_SIZE,
                                                            class_mode = None, seed = seed1, subset = 'validation',
                                                            color_mode = color_mode)
        
    valid_image_generator_1 = image_datagen.flow_from_directory(ALL_IMAGES.parent,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=VALID_BATCH_SIZE,
                                                              class_mode = None, seed = seed2, subset = 'validation')
    valid_mask_generator_1 = mask_datagen.flow_from_directory(ALL_MASKS.parent,
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=VALID_BATCH_SIZE,
                                                            class_mode = None, seed = seed2, subset = 'validation',
                                                            color_mode = color_mode)
                                                            
    valid_image_generator_2 = image_datagen.flow_from_directory(ALL_IMAGES.parent,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=VALID_BATCH_SIZE,
                                                              class_mode = None, seed = seed3, subset = 'validation')
    valid_mask_generator_2 = mask_datagen.flow_from_directory(ALL_MASKS.parent,
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=VALID_BATCH_SIZE,
                                                            class_mode = None, seed = seed3, subset = 'validation',
                                                            color_mode = color_mode)
                                                                                                                        
    train_generator = disjoin_generators((train_image_generator, train_mask_generator,train_image_generator_1, 
                                      train_mask_generator_1, train_image_generator_2, 
                                      train_mask_generator_2))
    valid_generator = disjoin_generators((valid_image_generator, valid_mask_generator,
                                     valid_image_generator_1, valid_mask_generator_1, train_image_generator_2, 
                                      train_mask_generator_2))                                                   
                                                            
    return train_generator, valid_generator       


                                 
def EnsembleNet(train_generator, valid_generator, s = 1, scheduler = 'Adam', compiler = 'Adam', n = 1, 
      map_weights = 'no', model_name = 'Ensemble.h5'):
    
    inputs_1 = Input((None, None, 3))
    inputs_2 = Input((None, None, 3))
    inputs_3 = Input((None, None, 3))
        
    c1_1 = Conv2D(8*n, (7, 7), padding='same',  kernel_initializer = 'he_normal')(inputs_1)
    c1_1 = BatchNormalization()(c1_1)
    c1_1 = Activation('elu')(c1_1)

    c1_1 = Conv2D(8*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c1_1)

    p1_1 = MaxPooling2D((2, 2))(c1_1)

    ##########################################
    X_shortcut = Conv2D(16*n, (1, 1), padding='same', kernel_initializer = 'he_normal')(p1_1)
    X_shortcut = BatchNormalization()(X_shortcut)

    c2_1 = BatchNormalization()(p1_1)
    c2_1 = Activation('elu')(c2_1)
    c2_1 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c2_1)

    c2_1 = BatchNormalization()(c2_1)
    c2_1 = Activation('elu')(c2_1)
    c2_1 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c2_1)

    c2_1 = Add()([c2_1, X_shortcut])

    p2_1 = MaxPooling2D((2, 2))(c2_1)

    ##########################################
    X_shortcut = Conv2D(32*n, (1, 1), padding='same', kernel_initializer = 'he_normal')(p2_1)
    X_shortcut = BatchNormalization()(X_shortcut)

    c3_1 = BatchNormalization()(p2_1)
    c3_1 = Activation('elu')(c3_1)
    c3_1 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c3_1)

    c3_1 = BatchNormalization()(c3_1)
    c3_1 = Activation('elu')(c3_1)
    c3_1 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c3_1)

    c3_1 = Add()([c3_1, X_shortcut])

    p3_1 = MaxPooling2D((2, 2))(c3_1)

    ###################BRIDGE FOR FIRST AND SECOND #######################

    ####################SECOND BRANCH##################

    c1_2 = Conv2D(8*n, (7, 7), padding='same',  kernel_initializer = 'he_normal')(inputs_2)
    c1_2 = BatchNormalization()(c1_2)
    c1_2 = Activation('elu')(c1_2)

    c1_2 = Conv2D(8*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c1_2)

    p1_2 = MaxPooling2D((2, 2))(c1_2)

    ##########################################
    X_shortcut = Conv2D(16*n, (1, 1), padding='same', kernel_initializer = 'he_normal')(p1_2)
    X_shortcut = BatchNormalization()(X_shortcut)

    c2_2 = BatchNormalization()(p1_2)
    c2_2 = Activation('elu')(c2_2)
    c2_2 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c2_2)

    c2_2 = BatchNormalization()(c2_2)
    c2_2 = Activation('elu')(c2_2)
    c2_2 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c2_2)

    c2_2 = Add()([c2_2, X_shortcut])

    p2_2 = MaxPooling2D((2, 2))(c2_2)

    ##########################################
    X_shortcut = Conv2D(32*n, (1, 1), padding='same', kernel_initializer = 'he_normal')(p2_2)
    X_shortcut = BatchNormalization()(X_shortcut)

    c3_2 = BatchNormalization()(p2_2)
    c3_2 = Activation('elu')(c3_2)
    c3_2 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c3_2)

    c3_2 = BatchNormalization()(c3_2)
    c3_2 = Activation('elu')(c3_2)
    c3_2 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c3_2)

    c3_2 = Add()([c3_2, X_shortcut])

    p3_2 = MaxPooling2D((2, 2))(c3_2)

    ################################################

    ####################THIRD BRANCH##################

    c1_3 = Conv2D(8*n, (7, 7), padding='same',  kernel_initializer = 'he_normal')(inputs_3)
    c1_3 = BatchNormalization()(c1_3)
    c1_3 = Activation('elu')(c1_3)

    c1_3 = Conv2D(8*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c1_3)

    p1_3 = MaxPooling2D((2, 2))(c1_3)

    ##########################################
    X_shortcut = Conv2D(16*n, (1, 1), padding='same', kernel_initializer = 'he_normal')(p1_3)
    X_shortcut = BatchNormalization()(X_shortcut)

    c2_3 = BatchNormalization()(p1_3)
    c2_3 = Activation('elu')(c2_3)
    c2_3 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c2_3)

    c2_3 = BatchNormalization()(c2_3)
    c2_3 = Activation('elu')(c2_3)
    c2_3 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c2_3)

    c2_3 = Add()([c2_3, X_shortcut])

    p2_3 = MaxPooling2D((2, 2))(c2_3)

    ##########################################
    X_shortcut = Conv2D(32*n, (1, 1), padding='same', kernel_initializer = 'he_normal')(p2_3)
    X_shortcut = BatchNormalization()(X_shortcut)

    c3_3 = BatchNormalization()(p2_3)
    c3_3 = Activation('elu')(c3_3)
    c3_3 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c3_3)

    c3_3 = BatchNormalization()(c3_3)
    c3_3 = Activation('elu')(c3_3)
    c3_3 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c3_3)

    c3_3 = Add()([c3_3, X_shortcut])

    p3_3 = MaxPooling2D((2, 2))(c3_3)

    #############FIRST BRIDGE ###############

    X_shortcut = Conv2D(64*n, (1, 1), padding='same', kernel_initializer = 'he_normal')(p3_1)
    X_shortcut = BatchNormalization()(X_shortcut)

    c4_1 = BatchNormalization()(p3_1)
    c4_1 = Activation('elu')(c4_1)
    c4_1 = Conv2D(64*n, (5, 5), padding='same',kernel_initializer = 'he_normal')(c4_1)

    c4_1 = BatchNormalization()(c4_1)
    c4_1 = Activation('elu')(c4_1)
    c4_1 = Conv2D(64*n, (5, 5), padding='same',kernel_initializer = 'he_normal')(c4_1)

    c5_1 = BatchNormalization()(c4_1)
    c5_1 = Activation('elu')(c5_1)
    c5_1 = Conv2D(64*n, (5, 5), padding='same',kernel_initializer = 'he_normal')(c5_1)

    c5_1 = BatchNormalization()(c5_1)
    c5_1 = Activation('elu')(c5_1)
    c5_1 = Conv2D(64*n, (5, 5), padding='same',kernel_initializer = 'he_normal')(c5_1)

    c5_1 = Add()([c5_1, X_shortcut])

    ###########SECOND BRIDGE##############
    X_shortcut = Conv2D(64*n, (1, 1), padding='same', kernel_initializer = 'he_normal')(p3_2)
    X_shortcut = BatchNormalization()(X_shortcut)

    c4_2 = BatchNormalization()(p3_2)
    c4_2 = Activation('elu')(c4_2)
    c4_2 = Conv2D(64*n, (5, 5), padding='same',kernel_initializer = 'he_normal')(c4_2)

    c4_2 = BatchNormalization()(c4_2)
    c4_2 = Activation('elu')(c4_2)
    c4_2 = Conv2D(64*n, (5, 5), padding='same',kernel_initializer = 'he_normal')(c4_2)

    c5_2 = BatchNormalization()(c4_2)
    c5_2 = Activation('elu')(c5_2)
    c5_2 = Conv2D(64*n, (5, 5), padding='same',kernel_initializer = 'he_normal')(c5_2)

    c5_2 = BatchNormalization()(c5_2)
    c5_2 = Activation('elu')(c5_2)
    c5_2 = Conv2D(64*n, (5, 5), padding='same',kernel_initializer = 'he_normal')(c5_2)

    c5_2 = Add()([c5_2, X_shortcut])

    ###################END BRIDGE#######################

    ###########STHIRD BRIDGE##############
    X_shortcut = Conv2D(64*n, (1, 1), padding='same', kernel_initializer = 'he_normal')(p3_3)
    X_shortcut = BatchNormalization()(X_shortcut)

    c4_3 = BatchNormalization()(p3_3)
    c4_3 = Activation('elu')(c4_3)
    c4_3 = Conv2D(64*n, (5, 5), padding='same',kernel_initializer = 'he_normal')(c4_3)

    c4_3 = BatchNormalization()(c4_3)
    c4_3 = Activation('elu')(c4_3)
    c4_3 = Conv2D(64*n, (5, 5), padding='same',kernel_initializer = 'he_normal')(c4_3)

    c5_3 = BatchNormalization()(c4_3)
    c5_3 = Activation('elu')(c5_3)
    c5_3 = Conv2D(64*n, (5, 5), padding='same',kernel_initializer = 'he_normal')(c5_3)

    c5_3 = BatchNormalization()(c5_3)
    c5_3 = Activation('elu')(c5_3)
    c5_3 = Conv2D(64*n, (5, 5), padding='same',kernel_initializer = 'he_normal')(c5_3)

    c5_3 = Add()([c5_3, X_shortcut])

    ############FIRST BRANCH##############

    u6_1 = Conv2DTranspose(32*n, (2, 2), strides=(2, 2), padding='same') (c5_1)
    u6_1 = concatenate([u6_1, c3_1])

    X_shortcut = Conv2D(32*n, (1, 1), padding='same', kernel_initializer = 'he_normal') (u6_1)
    X_shortcut = BatchNormalization()(X_shortcut)

    c6_1 = BatchNormalization()(u6_1)
    c6_1 = Activation('elu')(c6_1)
    c6_1 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c6_1)

    c6_1 = BatchNormalization()(c6_1)
    c6_1 = Activation('elu')(c6_1)
    c6 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c6_1)

    c6_1 = Add()([c6_1, X_shortcut])

    ################################################

    u7_1 = Conv2DTranspose(16*n, (2, 2), strides=(2, 2), padding='same') (c6_1)
    u7_1 = concatenate([u7_1, c2_1])

    X_shortcut = Conv2D(16*n, (1, 1), padding='same', kernel_initializer = 'he_normal') (u7_1)
    X_shortcut = BatchNormalization()(X_shortcut)

    c7_1 = BatchNormalization()(u7_1)
    c7_1 = Activation('elu')(c7_1)
    c7_1 = Conv2D(16*n, (3, 3), padding='same',kernel_initializer = 'he_normal')(c7_1)

    c7_1 = BatchNormalization()(c7_1)
    c7_1 = Activation('elu')(c7_1)
    c7_1 = Conv2D(16*n, (3, 3), padding='same',kernel_initializer = 'he_normal')(c7_1)

    c7_1 = Add()([c7_1, X_shortcut])

    ##########################################################

    u8_1 = Conv2DTranspose(8*n, (2, 2), strides=(2, 2), padding='same') (c7_1)
    u8_1 = concatenate([u8_1, c1_1])

    X_shortcut = Conv2D(8*n, (1, 1), padding='same', kernel_initializer = 'he_normal') (u8_1)
    X_shortcut = BatchNormalization()(X_shortcut)

    c8_1 = BatchNormalization()(u8_1)
    c8_1 = Activation('elu')(c8_1)
    c8_1 = Conv2D(8*n, (3, 3), padding='same',kernel_initializer = 'he_normal')(c8_1)

    c8_1 = BatchNormalization()(c8_1)
    c8_1 = Activation('elu')(c8_1)
    c8_1 = Conv2D(8*n, (3, 3), padding='same',kernel_initializer = 'he_normal')(c8_1)
    
    c8_1 = Add()([c8_1, X_shortcut])


    segmentation_1 = Conv2D(1, (1, 1), activation='sigmoid', name = 'first_model') (c8_1)

    ############SECONDO BRANCH##########

    u6_2 = Conv2DTranspose(32*n, (2, 2), strides=(2, 2), padding='same') (c5_2)
    u6_2 = concatenate([u6_2, c3_2])

    X_shortcut = Conv2D(32*n, (1, 1), padding='same', kernel_initializer = 'he_normal') (u6_2)
    X_shortcut = BatchNormalization()(X_shortcut)

    c6_2 = BatchNormalization()(u6_2)
    c6_2 = Activation('elu')(c6_2)
    c6_2 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c6_2)

    c6_2 = BatchNormalization()(c6_2)
    c6_2 = Activation('elu')(c6_2)
    c6_2 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c6_2)

    c6_2 = Add()([c6_2, X_shortcut])

    ################################################

    u7_2 = Conv2DTranspose(16*n, (2, 2), strides=(2, 2), padding='same') (c6_2)
    u7_2 = concatenate([u7_2, c2_2])

    X_shortcut = Conv2D(16*n, (1, 1), padding='same', kernel_initializer = 'he_normal') (u7_2)
    X_shortcut = BatchNormalization()(X_shortcut)

    c7_2 = BatchNormalization()(u7_2)
    c7_2 = Activation('elu')(c7_2)
    c7_2 = Conv2D(16*n, (3, 3), padding='same',kernel_initializer = 'he_normal')(c7_2)

    c7_2 = BatchNormalization()(c7_2)
    c7_2 = Activation('elu')(c7_2)
    c7_2 = Conv2D(16*n, (3, 3), padding='same',kernel_initializer = 'he_normal')(c7_2)

    c7_2 = Add()([c7_2, X_shortcut])

    #########################################################################

    u8_2 = Conv2DTranspose(8*n, (2, 2), strides=(2, 2), padding='same') (c7_2)
    u8_2 = concatenate([u8_2, c1_2])

    X_shortcut = Conv2D(8*n, (1, 1), padding='same', kernel_initializer = 'he_normal') (u8_2)
    X_shortcut = BatchNormalization()(X_shortcut)

    c8_2 = BatchNormalization()(u8_2)
    c8_2 = Activation('elu')(c8_2)
    c8_2 = Conv2D(8*n, (3, 3), padding='same',kernel_initializer = 'he_normal')(c8_2)

    c8_2 = BatchNormalization()(c8_2)
    c8_2 = Activation('elu')(c8_2)
    c8_2 = Conv2D(8*n, (3, 3), padding='same',kernel_initializer = 'he_normal')(c8_2)

    c8_2 = Add()([c8_2, X_shortcut])

    segmentation_2 = Conv2D(1, (1, 1), activation='sigmoid', name = 'second_model') (c8_2)

    ##########################THIRD BRANCH################################

    u6_3 = Conv2DTranspose(32*n, (2, 2), strides=(2, 2), padding='same') (c5_3)
    u6_3 = concatenate([u6_3, c3_3])

    X_shortcut = Conv2D(32*n, (1, 1), padding='same', kernel_initializer = 'he_normal') (u6_3)
    X_shortcut = BatchNormalization()(X_shortcut)

    c6_3 = BatchNormalization()(u6_3)
    c6_3 = Activation('elu')(c6_3)
    c6_3 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c6_3)

    c6_3 = BatchNormalization()(c6_3)
    c6_3 = Activation('elu')(c6_3)
    c6_3 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c6_3)

    c6_3 = Add()([c6_3, X_shortcut])

    ################################################

    u7_3 = Conv2DTranspose(16*n, (2, 2), strides=(2, 2), padding='same') (c6_3)
    u7_3 = concatenate([u7_3, c2_3])

    X_shortcut = Conv2D(16*n, (1, 1), padding='same', kernel_initializer = 'he_normal') (u7_3)
    X_shortcut = BatchNormalization()(X_shortcut)

    c7_3 = BatchNormalization()(u7_3)
    c7_3 = Activation('elu')(c7_3)
    c7_3 = Conv2D(16*n, (3, 3), padding='same',kernel_initializer = 'he_normal')(c7_3)

    c7_3 = BatchNormalization()(c7_3)
    c7_3 = Activation('elu')(c7_3)
    c7_3 = Conv2D(16*n, (3, 3), padding='same',kernel_initializer = 'he_normal')(c7_3)

    c7_3 = Add()([c7_3, X_shortcut])

    #######################################################

    u8_3 = Conv2DTranspose(8*n, (2, 2), strides=(2, 2), padding='same') (c7_3)
    u8_3 = concatenate([u8_3, c1_3])

    X_shortcut = Conv2D(8*n, (1, 1), padding='same', kernel_initializer = 'he_normal') (u8_3)
    X_shortcut = BatchNormalization()(X_shortcut)

    c8_3 = BatchNormalization()(u8_3)
    c8_3 = Activation('elu')(c8_3)
    c8_3 = Conv2D(8*n, (3, 3), padding='same',kernel_initializer = 'he_normal')(c8_3)

    c8_3 = BatchNormalization()(c8_3)
    c8_3 = Activation('elu')(c8_3)
    c8_3 = Conv2D(8*n, (3, 3), padding='same',kernel_initializer = 'he_normal')(c8_3)

    c8_3 = Add()([c8_3, X_shortcut])

    segmentation_3 = Conv2D(1, (1, 1), activation='sigmoid', name = 'third_model') (c8_3)

    model = Model(inputs=[inputs_1, inputs_2, inputs_3], outputs=[segmentation_1, segmentation_2, segmentation_3])
  
    model.summary()


    if map_weights == 'no':
    
        WeightedLoss = create_weighted_binary_crossentropy(1, 1.25)  
        
    elif map_weights == 'yes':

        WeightedLoss_1 = create_weighted_binary_crossentropy_2(1, 1) 
        WeightedLoss_2 = create_weighted_binary_crossentropy_2(1.15, 1)             
    
     

    if compiler == 'Adam':          
        Adam = optimizers.Adam(lr=0.0015)
        model.compile(optimizer=Adam, loss=[WeightedLoss_1, WeightedLoss_2, WeightedLoss_1], 
              metrics={'first_model':[mean_iou,dice_coef],
                      'second_model':[mean_iou,dice_coef],
                      'third_model':[mean_iou,dice_coef]
                      })

    
    #model = load_model(str(MODEL_CHECKPOINTS/'UnetSmall_v4.h5'), custom_objects={'mean_iou': mean_iou, 'dice_coef': dice_coef, 'weighted_binary_crossentropy': WeightedLoss})    
    
    checkpointer = ModelCheckpoint(str(MODEL_CHECKPOINTS/model_name), verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=30)
    tensorboard = tb.TrainValTensorBoard(log_dir=str(RESULTS_DIRECTORY / 'logs'), write_graph=True, write_images=True,
                                         batch_size = BATCH_SIZE, write_grads=False) 
    #tensorboard = TensorBoard(log_dir=str(RESULTS_DIRECTORY / 'logs'), histogram_freq=1, batch_size=BATCH_SIZE, write_graph=True, 
                              #write_grads=False, write_images=True, embeddings_freq=1, embeddings_layer_names=None, 
                              #embeddings_metadata=None, embeddings_data=None, update_freq='batch')
    plot_losses = plots.TrainingPlot()
    schedule = lrate.SGDRScheduler(min_lr=1e-4,
                             max_lr=1e-3,
                             steps_per_epoch=np.ceil(train_number/BATCH_SIZE),
                             lr_decay=0.8,
                             cycle_length=15,
                             mult_factor=1.25)

    ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=3, verbose=1, 
                             mode='auto', cooldown=0, min_lr=9e-5)
    if scheduler == 'Adam':          
        callbacks = [checkpointer, earlystopping, tensorboard, ReduceLR]
    elif scheduler == 'SGD':
        callbacks = [checkpointer, earlystopping, tensorboard, schedule]
        
    results = model.fit_generator(train_generator, 
                                  steps_per_epoch=(train_number/BATCH_SIZE)-1,
                                  validation_data=valid_generator, 
                                  validation_steps=(valid_number/VALID_BATCH_SIZE)-1, 
                                  callbacks=callbacks,
                                  epochs=200)



def EnsembleDeepResUnet(train_generator, valid_generator, s = 1, scheduler = 'Adam', compiler = 'Adam', n = 1, 
      map_weights = 'no', model_name = 'Ensemble.h5'):



    inputs_1 = Input((None, None, 3))
    inputs_2 = Input((None, None, 3))
        
    c1 = Conv2D(8*n, (7, 7), padding='same',  kernel_initializer = 'he_normal')(inputs_1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('elu')(c1)

    c1 = Conv2D(8*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c1)

    p1 = MaxPooling2D((2, 2))(c1)

    ##########################################
    X_shortcut = Conv2D(16*n, (1, 1), padding='same', kernel_initializer = 'he_normal')(p1)
    X_shortcut = BatchNormalization()(X_shortcut)

    c2 = BatchNormalization()(p1)
    c2 = Activation('elu')(c2)
    c2 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c2)

    c2 = BatchNormalization()(c2)
    c2 = Activation('elu')(c2)
    c2 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c2)

    c2 = Add()([c2, X_shortcut])

    p2 = MaxPooling2D((2, 2))(c2)

    ##########################################
    X_shortcut = Conv2D(32*n, (1, 1), padding='same', kernel_initializer = 'he_normal')(p2)
    X_shortcut = BatchNormalization()(X_shortcut)

    c3 = BatchNormalization()(p2)
    c3 = Activation('elu')(c3)
    c3 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c3)

    c3 = BatchNormalization()(c3)
    c3 = Activation('elu')(c3)
    c3 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c3)

    c3 = Add()([c3, X_shortcut])

    p3 = MaxPooling2D((2, 2))(c3)

    ####################SECOND BRANCH##################

    c1_2 = Conv2D(8*n, (7, 7), padding='same',  kernel_initializer = 'he_normal')(inputs_2)
    c1_2 = BatchNormalization()(c1_2)
    c1_2 = Activation('elu')(c1_2)

    c1_2 = Conv2D(8*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c1_2)

    p1_2 = MaxPooling2D((2, 2))(c1_2)

    ##########################################
    X_shortcut = Conv2D(16*n, (1, 1), padding='same', kernel_initializer = 'he_normal')(p1_2)
    X_shortcut = BatchNormalization()(X_shortcut)

    c2_2 = BatchNormalization()(p1_2)
    c2_2 = Activation('elu')(c2_2)
    c2_2 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c2_2)

    c2_2 = BatchNormalization()(c2_2)
    c2_2 = Activation('elu')(c2_2)
    c2_2 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c2_2)

    c2_2 = Add()([c2_2, X_shortcut])

    p2_2 = MaxPooling2D((2, 2))(c2_2)

    ##########################################
    X_shortcut = Conv2D(32*n, (1, 1), padding='same', kernel_initializer = 'he_normal')(p2_2)
    X_shortcut = BatchNormalization()(X_shortcut)

    c3_2 = BatchNormalization()(p2_2)
    c3_2 = Activation('elu')(c3_2)
    c3_2 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c3_2)

    c3_2 = BatchNormalization()(c3_2)
    c3_2 = Activation('elu')(c3_2)
    c3_2 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c3_2)

    c3_2 = Add()([c3_2, X_shortcut])

    p3_2 = MaxPooling2D((2, 2))(c3_2)

    ###################BRIDGE FOR FIRST AND SECOND #######################

    X_shortcut = Conv2D(64*n, (1, 1), padding='same', kernel_initializer = 'he_normal')(p3)
    X_shortcut = BatchNormalization()(X_shortcut)

    c4 = BatchNormalization()(p3)
    c4 = Activation('elu')(c4)
    c4 = Conv2D(64*n, (5, 5), padding='same',kernel_initializer = 'he_normal')(c4)

    c4 = BatchNormalization()(c4)
    c4 = Activation('elu')(c4)
    c4 = Conv2D(64*n, (5, 5), padding='same',kernel_initializer = 'he_normal')(c4)

    c5 = BatchNormalization()(c4)
    c5 = Activation('elu')(c5)
    c5 = Conv2D(64*n, (5, 5), padding='same',kernel_initializer = 'he_normal')(c5)

    c5 = BatchNormalization()(c5)
    c5 = Activation('elu')(c5)
    c5 = Conv2D(64*n, (5, 5), padding='same',kernel_initializer = 'he_normal')(c5)

    c5 = Add()([c5, X_shortcut])

    ###################END BRIDGE#######################

    ##############BRIDGE SECONDO BRANCH###################

    X_shortcut = Conv2D(64*n, (1, 1), padding='same', kernel_initializer='he_normal')(p3_2)

    c51 = BatchNormalization()(p3)
    c51 = Activation('elu')(c51)
    c51 = Conv2D(16*n, (3, 3),dilation_rate = (6,6),padding='same', kernel_initializer='he_normal')(c51)

    c52 = BatchNormalization()(p3)
    c52 = Activation('elu')(c52)
    c52 = Conv2D(16*n, (3, 3), dilation_rate = (12,12), padding='same', kernel_initializer='he_normal')(c52)

    c53 = BatchNormalization()(p3)
    c53 = Activation('elu')(c53)
    c53 = Conv2D(16*n, (3, 3), dilation_rate = (18,18), padding='same', kernel_initializer='he_normal')(c53)

    c54 = BatchNormalization()(p3)
    c54 = Activation('elu')(c54)
    c54 = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal')(c54)

    c5_2 = concatenate([c51,c52,c53,c54])

    c5_2 = Conv2D(64*n, (1, 1), padding='same', kernel_initializer='he_normal')(c5)

    c5_2 = Add()([c5, X_shortcut])


    ############FIRST BRANCH##############

    u6 = Conv2DTranspose(32*n, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c3])

    X_shortcut = Conv2D(32*n, (1, 1), padding='same', kernel_initializer = 'he_normal') (u6)
    X_shortcut = BatchNormalization()(X_shortcut)

    c6 = BatchNormalization()(u6)
    c6 = Activation('elu')(c6)
    c6 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c6)

    c6 = BatchNormalization()(c6)
    c6 = Activation('elu')(c6)
    c6 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c6)

    c6 = Add()([c6, X_shortcut])

    ################################################

    u7 = Conv2DTranspose(16*n, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c2])

    X_shortcut = Conv2D(16*n, (1, 1), padding='same', kernel_initializer = 'he_normal') (u7)
    X_shortcut = BatchNormalization()(X_shortcut)

    c7 = BatchNormalization()(u7)
    c7 = Activation('elu')(c7)
    c7 = Conv2D(16*n, (3, 3), padding='same',kernel_initializer = 'he_normal')(c7)

    c7 = BatchNormalization()(c7)
    c7 = Activation('elu')(c7)
    c7 = Conv2D(16*n, (3, 3), padding='same',kernel_initializer = 'he_normal')(c7)

    c7 = Add()([c7, X_shortcut])

    u8 = Conv2DTranspose(8*n, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c1])

    X_shortcut = Conv2D(8*n, (1, 1), padding='same', kernel_initializer = 'he_normal') (u8)
    X_shortcut = BatchNormalization()(X_shortcut)

    c8 = BatchNormalization()(u8)
    c8 = Activation('elu')(c8)
    c8 = Conv2D(8*n, (3, 3), padding='same',kernel_initializer = 'he_normal')(c8)

    c8 = BatchNormalization()(c8)
    c8 = Activation('elu')(c8)
    c8 = Conv2D(8*n, (3, 3), padding='same',kernel_initializer = 'he_normal')(c8)


    segmentation_1 = Conv2D(1, (1, 1), activation='sigmoid', name = 'first_model') (c8)

    ############SECONDO BRANCH##########

    u6_2 = UpSampling2D(interpolation='bilinear')(c5_2)
    u6_2 = concatenate([u6_2, c3])

    X_shortcut = Conv2D(32*n, (1, 1), padding='same', kernel_initializer = 'he_normal') (u6_2)
    X_shortcut = BatchNormalization()(X_shortcut)

    c6_2 = BatchNormalization()(u6_2)
    c6_2 = Activation('elu')(c6_2)
    c6_2 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c6_2)

    c6_2 = BatchNormalization()(c6_2)
    c6_2 = Activation('elu')(c6_2)
    c6_2 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer = 'he_normal')(c6_2)

    c6_2 = Add()([c6_2, X_shortcut])

    ################################################

    u7_2 =UpSampling2D(interpolation='bilinear')(c6_2)
    u7_2 = concatenate([u7_2, c2])

    X_shortcut = Conv2D(16*n, (1, 1), padding='same', kernel_initializer = 'he_normal') (u7_2)
    X_shortcut = BatchNormalization()(X_shortcut)

    c7_2 = BatchNormalization()(u7_2)
    c7_2 = Activation('elu')(c7_2)
    c7_2 = Conv2D(16*n, (3, 3), padding='same',kernel_initializer = 'he_normal')(c7_2)

    c7_2 = BatchNormalization()(c7_2)
    c7_2 = Activation('elu')(c7_2)
    c7_2 = Conv2D(16*n, (3, 3), padding='same',kernel_initializer = 'he_normal')(c7_2)

    c7_2 = Add()([c7_2, X_shortcut])

    #########################################################################

    u8_2 = UpSampling2D(interpolation='bilinear')(c7_2)
    u8_2 = concatenate([u8_2, c1])

    X_shortcut = Conv2D(8*n, (1, 1), padding='same', kernel_initializer = 'he_normal') (u8_2)
    X_shortcut = BatchNormalization()(X_shortcut)

    c8_2 = BatchNormalization()(u8_2)
    c8_2 = Activation('elu')(c8_2)
    c8_2 = Conv2D(8*n, (3, 3), padding='same',kernel_initializer = 'he_normal')(c8_2)

    c8_2 = BatchNormalization()(c8_2)
    c8_2 = Activation('elu')(c8_2)
    c8_2 = Conv2D(8*n, (3, 3), padding='same',kernel_initializer = 'he_normal')(c8_2)

    c8_2 = Add()([c8_2, X_shortcut])

    segmentation_2 = Conv2D(1, (1, 1), activation='sigmoid', name = 'second_model') (c8_2)

    model = Model(inputs=[inputs_1,inputs_2], outputs=[segmentation_1, segmentation_2])
      
      
    model.summary()

    if map_weights == 'no':
    
        WeightedLoss = create_weighted_binary_crossentropy(1, 1.25)  
        
    elif map_weights == 'yes':

        WeightedLoss_1 = create_weighted_binary_crossentropy_2(1.0, 1)
        WeightedLoss_2 = create_weighted_binary_crossentropy_2(1.2, 1)        
    
     

    if compiler == 'Adam':          
        Adam = optimizers.Adam(lr=0.0015)
        model.compile(optimizer=Adam, loss=[WeightedLoss_1, WeightedLoss_2], 
              metrics={'first_model':[mean_iou,dice_coef],
                      'second_model':[mean_iou,dice_coef]
                      })

    
    #model = load_model(str(MODEL_CHECKPOINTS/'UnetSmall_v4.h5'), custom_objects={'mean_iou': mean_iou, 'dice_coef': dice_coef, 'weighted_binary_crossentropy': WeightedLoss})    
    
    checkpointer = ModelCheckpoint(str(MODEL_CHECKPOINTS/model_name), verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=30)
    tensorboard = tb.TrainValTensorBoard(log_dir=str(RESULTS_DIRECTORY / 'logs'), write_graph=True, write_images=True,
                                         batch_size = BATCH_SIZE, write_grads=False) 
    #tensorboard = TensorBoard(log_dir=str(RESULTS_DIRECTORY / 'logs'), histogram_freq=1, batch_size=BATCH_SIZE, write_graph=True, 
                              #write_grads=False, write_images=True, embeddings_freq=1, embeddings_layer_names=None, 
                              #embeddings_metadata=None, embeddings_data=None, update_freq='batch')
    plot_losses = plots.TrainingPlot()
    schedule = lrate.SGDRScheduler(min_lr=1e-4,
                             max_lr=1e-3,
                             steps_per_epoch=np.ceil(train_number/BATCH_SIZE),
                             lr_decay=0.8,
                             cycle_length=15,
                             mult_factor=1.25)

    ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=3, verbose=1, 
                             mode='auto', cooldown=0, min_lr=9e-5)
    if scheduler == 'Adam':          
        callbacks = [checkpointer, earlystopping, tensorboard, ReduceLR]
    elif scheduler == 'SGD':
        callbacks = [checkpointer, earlystopping, tensorboard, schedule]
        
    results = model.fit_generator(train_generator, 
                                  steps_per_epoch=train_number/BATCH_SIZE,
                                  validation_data=valid_generator, 
                                  validation_steps=valid_number/VALID_BATCH_SIZE, 
                                  callbacks=callbacks,
                                  epochs=200)
                                  


if __name__ == "__main__":


    #train_generator, valid_generator = imageGenerator(color_mode='rgb', n_outputs = 2)
    train_generator, valid_generator = imageDisjointGenerator(color_mode='rgb')
    EnsembleNet(train_generator,valid_generator, s=1, scheduler = 'Adam', compiler='Adam',
                      n = 1, map_weights = 'yes', model_name = '3N1etAtlas.h5')
    