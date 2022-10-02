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
import random

from importlib import import_module
plots = import_module('051_plotscalars')
tb = import_module('050_tensorboard')
lrate = import_module('043_learning_rate')

from config_script import *

tot_img_after_aug = count_files_in_directory(0, dir_list = ALL_IMAGES)

BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
# seed1=1 
seed1 = 333
random.seed(seed1)
val_percentage = 0.25
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
        
        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true[:,:,:,0:1], y_pred[:,:,:,0:1])
    

        # Apply the weights
        weight_vector0 = y_true[:,:,:,0:1] * one_weight + (1. - y_true[:,:,:,0:1]) * zero_weight

        weight_vector = weight_vector0 * y_true[:,:,:,1:2]
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy
	
def create_weighted_binary_crossentropy_3(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):
    
        weights = y_true[:,:,:,1:2]
        
        weights *= 6.224407

        b_ce = K.binary_crossentropy(y_true[:,:,:,0:1], y_pred[:,:,:,0:1])

        # Apply the weights
        weight_class = y_true[:,:,:,0:1] * one_weight + (1. - y_true[:,:,:,0:1]) * zero_weight

        weights_vector = weight_class + weights
        weighted_b_ce = weights_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy
    
def create_weighted_binary_crossentropy_sum(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):

        b_ce = K.binary_crossentropy(y_true[:,:,:,0:1], y_pred[:,:,:,0:1])

        # Apply the weights
        weight_class = y_true[:,:,:,0:1] * one_weight  + (1. - y_true[:,:,:,0:1]) * (zero_weight-1)
        
        weights_pixel = y_true[:,:,:,1:2] - y_true[:,:,:,1:2]*y_true[:,:,:,0:1]
        weights_pixel *= 6.224407

        weight_vector = weight_class + weights_pixel
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy

def rgb2hsv(image):

    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(image)
    image = cv2.merge([h*1./358, s*1./255, v*1./255])
    return image

def imageGenerator(color_mode = 'grayscale'):

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
    
    train_generator = zip(train_image_generator,train_mask_generator)
    valid_generator = zip(valid_image_generator,valid_mask_generator)
    return train_generator, valid_generator                          
                                  


                           
                                  
def Resuneption(train_generator,valid_generator, scheduler = 'Adam',compiler='Adam', n = 1, map_weights = 'yes', 
                   model_name = 'InceptionUnet.h5'):
                   
    inputs = Input((None, None, 3))
    
    c1 = Conv2D(16*n, (7, 7), padding='same',  kernel_initializer='he_normal')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation('elu')(c1)

    c1 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c1)

    ##########################################
    X_shortcut = Conv2D(32*n, (1, 1), strides=(2,2),padding='same', kernel_initializer='he_normal')(c1)
    X_shortcut = BatchNormalization()(X_shortcut)

    c21 = BatchNormalization()(c1)
    c21 = Activation('elu')(c21)
    c21 = Conv2D(8*n, (1, 1), padding='same', kernel_initializer='he_normal')(c21)
    
    c21 = BatchNormalization()(c21)
    c21 = Activation('elu')(c21)
    c21 = Conv2D(8*n, (3, 3),strides=(2,2), padding='same', kernel_initializer='he_normal')(c21)
    
    c22 = BatchNormalization()(c1)
    c22 = Activation('elu')(c22)
    c22 = Conv2D(8*n, (1, 1), padding='same', kernel_initializer='he_normal')(c22)
    
    c22 = BatchNormalization()(c22)
    c22 = Activation('elu')(c22)
    c22 = Conv2D(8*n, (3, 3), padding='same', kernel_initializer='he_normal')(c22)
    
    c22 = BatchNormalization()(c22)
    c22 = Activation('elu')(c22)
    c22 = Conv2D(8*n, (3, 3),strides=(2,2), padding='same', kernel_initializer='he_normal')(c22)
    
    c23 = BatchNormalization()(c1)
    c23 = Activation('elu')(c23)
    c23 = Conv2D(8*n, (1, 1),strides=(2,2), padding='same', kernel_initializer='he_normal')(c23)
    
    p2 = MaxPooling2D((2, 2))(c1)
    c24 = BatchNormalization()(p2)
    c24 = Activation('elu')(c24)
    c24 = Conv2D(8*n, (1, 1), padding='same', kernel_initializer='he_normal')(c24)
    
    c2 = concatenate([c21,c22,c23,c24])
    c2 = Add()([c2, X_shortcut])

    #########################################################################
    
    X_shortcut = Conv2D(64*n, (1, 1),strides=(2,2), padding='same', kernel_initializer='he_normal')(c2)
    X_shortcut = BatchNormalization()(X_shortcut)

    c31 = BatchNormalization()(c2)
    c31 = Activation('elu')(c31)
    c31 = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal')(c31)
    
    c31 = BatchNormalization()(c31)
    c31 = Activation('elu')(c31)
    c31 = Conv2D(16*n, (3, 3),strides=(2,2), padding='same', kernel_initializer='he_normal')(c31)
    
    c32 = BatchNormalization()(c2)
    c32 = Activation('elu')(c32)
    c32 = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal')(c32)
    
    c32 = BatchNormalization()(c32)
    c32 = Activation('elu')(c32)
    c32 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c32)
    
    c32 = BatchNormalization()(c32)
    c32 = Activation('elu')(c32)
    c32 = Conv2D(16*n, (3, 3),strides=(2,2), padding='same', kernel_initializer='he_normal')(c32)
    
    c33 = BatchNormalization()(c2)
    c33 = Activation('elu')(c33)
    c33 = Conv2D(16*n, (1, 1),strides=(2,2), padding='same', kernel_initializer='he_normal')(c33)
    
    p3 = MaxPooling2D((2, 2))(c2)
    c34 = BatchNormalization()(p3)
    c34 = Activation('elu')(c34)
    c34 = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal')(c34)   
    
    c3 = concatenate([c31,c32,c33,c34])    
    c3 = Add()([c3, X_shortcut])
    
    ###################Bridge#######################
    
    X_shortcut = Conv2D(128*n, (1, 1),strides=(2,2), padding='same', kernel_initializer='he_normal')(c3)
    X_shortcut = BatchNormalization()(X_shortcut)

    c41 = BatchNormalization()(c3)
    c41 = Activation('elu')(c41)
    c41 = Conv2D(32*n, (1, 1), padding='same', kernel_initializer='he_normal')(c41)
    
    c41 = BatchNormalization()(c41)
    c41 = Activation('elu')(c41)
    c41 = Conv2D(32*n, (3, 3),strides=(2,2), padding='same', kernel_initializer='he_normal')(c41)
    
    c42 = BatchNormalization()(c3)
    c42 = Activation('elu')(c42)
    c42 = Conv2D(32*n, (1, 1), padding='same', kernel_initializer='he_normal')(c42)
    
    c42 = BatchNormalization()(c42)
    c42 = Activation('elu')(c42)
    c42 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer='he_normal')(c42)
    
    c42 = BatchNormalization()(c42)
    c42 = Activation('elu')(c42)
    c42 = Conv2D(32*n, (3, 3),strides=(2,2), padding='same', kernel_initializer='he_normal')(c42)
    
    c43 = BatchNormalization()(c3)
    c43 = Activation('elu')(c43)
    c43 = Conv2D(32*n, (1, 1),strides=(2,2), padding='same', kernel_initializer='he_normal')(c43)
    
    p4 = MaxPooling2D(pool_size=(2, 2))(c3)
    c44 = BatchNormalization()(p4)
    c44 = Activation('elu')(c44)
    c44 = Conv2D(32*n, (1, 1), padding='same', kernel_initializer='he_normal')(c44)
    
    c4 = concatenate([c41,c42,c43,c44])  
    c4 = Add()([c4, X_shortcut])
    
  ##############BRIDGE#################
   
    X_shortcut = Conv2D(256*n, (1, 1), padding='same', kernel_initializer='he_normal')(c4)
    X_shortcut = BatchNormalization()(X_shortcut)
    
    c5 = BatchNormalization()(c4)
    c5 = Activation('elu')(c5)
    c5 = Conv2D(256*n, (5, 5), padding='same',kernel_initializer='he_normal')(c5)

    c5 = BatchNormalization()(c5)
    c5 = Activation('elu')(c5)
    c5 = Conv2D(256*n, (5, 5), padding='same',kernel_initializer='he_normal')(c5)
    
    c51 = BatchNormalization()(c5)
    c51 = Activation('elu')(c51)
    c51 = Conv2D(256*n, (5, 5), padding='same',kernel_initializer='he_normal')(c51)

    c51 = BatchNormalization()(c51)
    c51 = Activation('elu')(c51)
    c51 = Conv2D(256*n, (5, 5), padding='same',kernel_initializer='he_normal')(c51)

    c51 = Add()([c5, X_shortcut])
    
    ###################END BRIDGE#######################

#     u6 = Conv2DTranspose(64*n, (2, 2), strides=(2, 2), padding='same') (c51)
    u6 = UpSampling2D((2,2), interpolation='bilinear')(c51)
    u6 = concatenate([u6, c3])
    
    X_shortcut = Conv2D(64*n, (1, 1), padding='same', kernel_initializer='he_normal') (u6)
    X_shortcut = BatchNormalization()(X_shortcut)

    c6 = BatchNormalization()(u6)
    c6 = Activation('elu')(c6)
    c6 = Conv2D(64*n, (3, 3), padding='same', kernel_initializer='he_normal')(c6)

    c6 = BatchNormalization()(c6)
    c6 = Activation('elu')(c6)
    c6 = Conv2D(64*n, (3, 3), padding='same', kernel_initializer='he_normal')(c6)

    c6 = Add()([c6, X_shortcut])

    ################################################

#     u7 = Conv2DTranspose(32*n, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = UpSampling2D((2,2), interpolation='bilinear')(c6)
    u7 = concatenate([u7, c2])
    
    X_shortcut = Conv2D(32*n, (1, 1), padding='same', kernel_initializer='he_normal') (u7)
    X_shortcut = BatchNormalization()(X_shortcut)

    c7 = BatchNormalization()(u7)
    c7 = Activation('elu')(c7)
    c7 = Conv2D(32*n, (3, 3), padding='same',kernel_initializer='he_normal')(c7)

    c7 = BatchNormalization()(c7)
    c7 = Activation('elu')(c7)
    c7 = Conv2D(32*n, (3, 3), padding='same',kernel_initializer='he_normal')(c7)

    c7 = Add()([c7, X_shortcut])

#     u8 = Conv2DTranspose(16*n, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = UpSampling2D((2,2), interpolation='bilinear')(c7)
    u8 = concatenate([u8, c1])
    
    X_shortcut = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal') (u8)
    X_shortcut = BatchNormalization()(X_shortcut)

    c8 = BatchNormalization()(u8)
    c8 = Activation('elu')(c8)
    c8 = Conv2D(16*n, (3, 3), padding='same',kernel_initializer='he_normal')(c8)

    c8 = BatchNormalization()(c8)
    c8 = Activation('elu')(c8)
    c8 = Conv2D(16*n, (3, 3), padding='same',kernel_initializer='he_normal')(c8)

    c8 = Add()([c8, X_shortcut])

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c8)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.summary() 
    
    
    if map_weights == 'no':
    
        WeightedLoss = create_weighted_binary_crossentropy(1, 1.25)  
        
    elif map_weights == 'yes':

        WeightedLoss = create_weighted_binary_crossentropy_2(1.05, 1)     

    
    if compiler == 'Adam':          
        Adam = optimizers.Adam(lr=0.001)
        model.compile(optimizer=Adam, loss=WeightedLoss, metrics=[mean_iou, dice_coef])
    elif compiler == 'SGD':
        SGD = optimizers.SGD(lr=0.002, momentum = 0.9)
        model.compile(optimizer=SGD, loss=WeightedLoss, metrics=[mean_iou, dice_coef])
    
    # model = load_model(str(MODEL_CHECKPOINTS/model_name),
    # custom_objects={'mean_iou': mean_iou, 'dice_coef': dice_coef, 'weighted_binary_crossentropy': WeightedLoss})    
    
    checkpointer = ModelCheckpoint(str(MODEL_CHECKPOINTS/model_name), verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=30)
    tensorboard = tb.TrainValTensorBoard(log_dir=str(RESULTS_DIRECTORY / 'logs'), write_graph=True, write_images=True,
                                         batch_size = BATCH_SIZE, write_grads=False) 
    #tensorboard = TensorBoard(log_dir=str(RESULTS_DIRECTORY / 'logs'), histogram_freq=1, batch_size=BATCH_SIZE, write_graph=True, 
                              #write_grads=False, write_images=True, embeddings_freq=1, embeddings_layer_names=None, 
                              #embeddings_metadata=None, embeddings_data=None, update_freq='batch')
    plot_losses = plots.TrainingPlot()
    schedule = lrate.SGDRScheduler(min_lr=1e-4,
                             max_lr=2e-3,
                             steps_per_epoch=np.ceil(train_number/BATCH_SIZE),
                             lr_decay=0.9,
                             cycle_length=15,
                             mult_factor=1.25)

    ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=3, verbose=1, 
                             mode='auto', cooldown=0, min_lr=3e-8)
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
                                  

def ResUnet(train_generator, valid_generator, scheduler = 'Adam', compiler = 'Adam', n = 1, 
      map_weights = 'no', model_name = 'ResUnetIdentity.h5', LoadModel='False'):
    
    inputs = Input((None, None, 3))
    
    # c0 = Conv2D(1, (1, 1), padding='same',  kernel_initializer='he_normal')(inputs)
    # c0 = BatchNormalization()(c0)
    # c0 = Activation('elu')(c0)
    
    c1 = Conv2D(4*n, (7, 7), padding='same',  kernel_initializer='he_normal')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation('elu')(c1)

    c1 = Conv2D(4*n, (3, 3), padding='same', kernel_initializer='he_normal')(c1)

    p1 = MaxPooling2D((2, 2))(c1)

    ##########################################
    
    X_shortcut = Conv2D(8*n, (1, 1), padding='same', kernel_initializer='he_normal')(p1)
    # X_shortcut = BatchNormalization()(X_shortcut)
    

    c2 = BatchNormalization()(p1)
    c2 = Activation('elu')(c2)
    c2 = Conv2D(8*n, (3, 3), padding='same', kernel_initializer='he_normal')(c2)

    c2 = BatchNormalization()(c2)
    c2 = Activation('elu')(c2)
    c2 = Conv2D(8*n, (3, 3), padding='same', kernel_initializer='he_normal')(c2)

    c2 = Add()([c2, X_shortcut])

    p2 = MaxPooling2D((2, 2))(c2)

    ##########################################

    X_shortcut = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal')(p2)
    # X_shortcut = BatchNormalization()(X_shortcut)
    
    c3 = BatchNormalization()(p2)
    c3 = Activation('elu')(c3)
    c3 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c3)

    c3 = BatchNormalization()(c3)
    c3 = Activation('elu')(c3)
    c3 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c3)

    c3 = Add()([c3, X_shortcut])

    p3 = MaxPooling2D((2, 2))(c3)

    ###################Bridge#######################
  
    X_shortcut = Conv2D(32*n, (1, 1), padding='same', kernel_initializer='he_normal')(p3)
    # X_shortcut = BatchNormalization()(X_shortcut)
    
    c4 = BatchNormalization()(p3)
    c4 = Activation('elu')(c4)
    c4 = Conv2D(32*n, (5, 5), padding='same',kernel_initializer='he_normal')(c4)

    c4 = BatchNormalization()(c4)
    c4 = Activation('elu')(c4)
    c4 = Conv2D(32*n, (5, 5), padding='same',kernel_initializer='he_normal')(c4)
      
    c4 = Add()([c4, X_shortcut])
    X_shortcut = c4
    
    c5 = BatchNormalization()(c4)
    c5 = Activation('elu')(c5)
    c5 = Conv2D(32*n, (5, 5), padding='same',kernel_initializer='he_normal')(c5)

    c5 = BatchNormalization()(c5)
    c5 = Activation('elu')(c5)
    c5 = Conv2D(32*n, (5, 5), padding='same',kernel_initializer='he_normal')(c5)

    c5 = Add()([c5, X_shortcut])

    ###################END BRIDGE#######################

    X_shortcut = Conv2DTranspose(16*n, (2, 2), strides=(2, 2), padding='same') (c5)
    # X_shortcut = BatchNormalization()(X_shortcut)
    u6 = concatenate([X_shortcut, c3])
    
    # u6 = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal')(u6)

    c6 = BatchNormalization()(u6)
    c6 = Activation('elu')(c6)
    c6 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c6)

    c6 = BatchNormalization()(c6)
    c6 = Activation('elu')(c6)
    c6 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c6)

    c6 = Add()([c6, X_shortcut])

    ################################################
    
    X_shortcut = Conv2DTranspose(8*n, (2, 2), strides=(2, 2), padding='same') (c6)
    # X_shortcut = BatchNormalization()(X_shortcut)
    u7 = concatenate([X_shortcut, c2])
    
    # u7 = Conv2D(8*n, (1, 1), padding='same', kernel_initializer='he_normal')(u7)

    c7 = BatchNormalization()(u7)
    c7 = Activation('elu')(c7)
    c7 = Conv2D(8*n, (3, 3), padding='same',kernel_initializer='he_normal')(c7)

    c7 = BatchNormalization()(c7)
    c7 = Activation('elu')(c7)
    c7 = Conv2D(8*n, (3, 3), padding='same',kernel_initializer='he_normal')(c7)

    c7 = Add()([c7, X_shortcut])
    
    X_shortcut = Conv2DTranspose(4*n, (2, 2), strides=(2, 2), padding='same') (c7)
    # X_shortcut = BatchNormalization()(X_shortcut)
    u8 = concatenate([X_shortcut, c1])
    
    # u8 = Conv2D(4*n, (1, 1), padding='same', kernel_initializer='he_normal')(u8

    c8 = BatchNormalization()(u8)
    c8 = Activation('elu')(c8)
    c8 = Conv2D(4*n, (3, 3), padding='same',kernel_initializer='he_normal')(c8)

    c8 = BatchNormalization()(c8)
    c8 = Activation('elu')(c8)
    c8 = Conv2D(4*n, (3, 3), padding='same',kernel_initializer='he_normal')(c8)

    c8 = Add()([c8, X_shortcut])

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c8)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    model.summary()


    if map_weights == 'no':
    
        WeightedLoss = create_weighted_binary_crossentropy(1, 1.5)  
        
    elif map_weights == 'yes':

        WeightedLoss = create_weighted_binary_crossentropy_2(1, 1.05)     
    

    if LoadModel == True: 
       model = load_model(str(MODEL_CHECKPOINTS/'ResUnetHaircutYellowN16_Relu.h5'), 
       custom_objects={'mean_iou': mean_iou, 'dice_coef': dice_coef, 'weighted_binary_crossentropy': WeightedLoss})    
       K.set_value(model.optimizer.lr, 0.001)

    elif compiler == 'Adam':          
        Adam = optimizers.Adam(lr=0.001)
        model.compile(optimizer=Adam, loss=WeightedLoss, metrics=[mean_iou, dice_coef])
    elif compiler == 'SGD':
        SGD = optimizers.SGD(lr=0.003, momentum = 0.9)
        model.compile(optimizer=SGD, loss=WeightedLoss, metrics=[mean_iou, dice_coef])

    
    checkpointer = ModelCheckpoint(str(MODEL_CHECKPOINTS/model_name), verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=30)
    tensorboard = tb.TrainValTensorBoard(log_dir=str(RESULTS_DIRECTORY / 'logs'), write_graph=True, write_images=True,
                                         batch_size = BATCH_SIZE, write_grads=False) 
    #tensorboard = TensorBoard(log_dir=str(RESULTS_DIRECTORY / 'logs'), histogram_freq=1, batch_size=BATCH_SIZE, write_graph=True, 
                              #write_grads=False, write_images=True, embeddings_freq=1, embeddings_layer_names=None, 
                              #embeddings_metadata=None, embeddings_data=None, update_freq='batch')
    plot_losses = plots.TrainingPlot()
    schedule = lrate.SGDRScheduler(min_lr=3e-4,
                             max_lr=3e-3,
                             steps_per_epoch=np.ceil(train_number/BATCH_SIZE),
                             lr_decay=0.5,
                             cycle_length=20,
                             mult_factor=1.25)

    ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, verbose=1, 
                             mode='auto', cooldown=0, min_lr=9e-8)
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


def DeepLab(train_generator, valid_generator, scheduler = 'Adam', compiler = 'Adam', n = 4, n1=2,
      map_weights = 'no', model_name = 'ResUnetIdentity.h5'):
      
    inputs = Input((None, None, 3))      
    
    # c0 = Conv2D(4*n, (1, 1), padding='same',  kernel_initializer='he_normal')(inputs)
    # c0 = BatchNormalization()(c0)
    # c0 = Activation('elu')(c0)
    
    c1 = Conv2D(4*n, (7, 7), padding='same',  kernel_initializer='he_normal')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)

    c1 = Conv2D(4*n, (3, 3), padding='same', kernel_initializer='he_normal')(c1)

    p1 = MaxPooling2D((2, 2))(c1)

    ##########################################
    
    X_shortcut = Conv2D(8*n, (1, 1), padding='same', kernel_initializer='he_normal')(p1)

    c2 = BatchNormalization()(p1)
    c2 = Activation('relu')(c2)
    c2 = Conv2D(8*n, (3, 3), padding='same', kernel_initializer='he_normal')(c2)

    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    c2 = Conv2D(8*n, (3, 3), padding='same', kernel_initializer='he_normal')(c2)

    c2 = Add()([c2, X_shortcut])

    p2 = MaxPooling2D((2, 2))(c2)

    ##########################################

    X_shortcut = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal')(p2)
    
    c3 = BatchNormalization()(p2)
    c3 = Activation('relu')(c3)
    c3 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c3)

    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    c3 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c3)

    c3 = Add()([c3, X_shortcut])

    p3 = MaxPooling2D((2, 2))(c3)

    
    ##########################################
    X_shortcut = Conv2D(32*n, (1, 1), padding='same', kernel_initializer='he_normal')(p3)
    
    c4 = BatchNormalization()(p3)
    c4 = Activation('relu')(c4)
    c4 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer='he_normal')(c4)

    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    c4 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer='he_normal')(c4)

    c4 = Add()([c4, X_shortcut])

    p4 = MaxPooling2D((2, 2))(c4)
    
    ###################Bridge####################### 

    X_shortcut = Conv2D(64*n, (1, 1), padding='same', kernel_initializer='he_normal')(p4)    
    
    c40 = BatchNormalization()(p4)
    c40 = Activation('relu')(c40)
    c40 = Conv2D(16*n, (1, 1), padding='same',kernel_initializer='he_normal')(c40)

    c41 = BatchNormalization()(p4)
    c41 = Activation('relu')(c41)
    c41 = Conv2D(16*n, (3, 3), dilation_rate=(6,6), padding='same',kernel_initializer='he_normal')(c41)
    
    c50 = BatchNormalization()(p4)
    c50 = Activation('relu')(c50)
    c50 = Conv2D(16*n, (3, 3), dilation_rate=(12,12),padding='same',kernel_initializer='he_normal')(c50)

    c51 = BatchNormalization()(p4)
    c51 = Activation('relu')(c51)
    c51 = Conv2D(16*n, (3, 3), dilation_rate=(18,18),padding='same',kernel_initializer='he_normal')(c51)
    
    c5 = concatenate([p4, c40, c41, c50, c51])   
    c5 = Conv2D(64*n, (1, 1), padding='same', kernel_initializer='he_normal')(c5)
    
    c5 = Add()([c5, X_shortcut])
    ###################END BRIDGE#######################

    u60 = Conv2DTranspose(16*n, (2, 2), strides=(4, 4), padding='same') (c5)
    u6 = concatenate([u60, c3])
    
    X_shortcut = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal')(u60)

    c6 = BatchNormalization()(u6)
    c6 = Activation('relu')(c6)
    c6 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c6)

    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)
    c6 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c6)

    c6 = Add()([c6, X_shortcut])

    ################################################

    u70 = Conv2DTranspose(4*n, (2, 2), strides=(4, 4), padding='same') (c6)
    u7 = concatenate([u70, c1])
    
    X_shortcut = Conv2D(4*n, (1, 1), padding='same', kernel_initializer='he_normal')(u70)

    c7 = BatchNormalization()(u7)
    c7 = Activation('relu')(c7)
    c7 = Conv2D(4*n, (3, 3), padding='same',kernel_initializer='he_normal')(c7)

    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)
    c7 = Conv2D(4*n, (3, 3), padding='same',kernel_initializer='he_normal')(c7)

    c7 = Add()([c7, X_shortcut])
    
   ################################################

    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c7)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    model.summary()

    if map_weights == 'no':
    
        WeightedLoss = create_weighted_binary_crossentropy(1, 1.35)  
        
    elif map_weights == 'yes':

        WeightedLoss = create_weighted_binary_crossentropy_2(1, 1)     
    
     

    if compiler == 'Adam':          
        Adam = optimizers.Adam(lr=0.003)
        model.compile(optimizer=Adam, loss=WeightedLoss, metrics=[mean_iou, dice_coef])
    elif compiler == 'SGD':
        SGD = optimizers.SGD(lr=0.003, momentum = 0.9)
        model.compile(optimizer=SGD, loss=WeightedLoss, metrics=[mean_iou, dice_coef])

    
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

    ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=3, verbose=1, 
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


    train_generator, valid_generator = imageGenerator(color_mode='rgb')

    ResUnet(train_generator,valid_generator, scheduler = 'Adam', compiler='Adam',
                      n = 4, map_weights = 'yes', model_name = 'ResUnetNewW_2.h5')
