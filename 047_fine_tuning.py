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
seed1=1 
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
    
def create_weighted_binary_crossentropy_3(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):
    
        weights = y_true[:,:,:,1:2]

        b_ce = K.binary_crossentropy(y_true[:,:,:,0:1], y_pred[:,:,:,0:1])

        # Apply the weights
        weight_vector0 = y_true[:,:,:,0:1] * one_weight + (1. - y_true[:,:,:,0:1]) * zero_weight
        
        weights *= 6.2812757

        weight_vector = weight_vector0 + weights
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
     

def FineTuning(train_generator,valid_generator,backbone_name='inceptionv3', scheduler = 'Adam', compiler='Adam',
                map_weights = 'no', backbone_model_name = 'FreezeInceptionTrainUnet.h5', fine_model_name = 'TrainIncTrainUnet.h5' ,
                full_train = True):

    if full_train:
        model = Unet(backbone_name=backbone_name, encoder_weights='imagenet', input_shape=(None,None,3), freeze_encoder=True)
        preprocess_input = get_preprocessing(backbone_name)
        model_name_1 = 'FreezeInceptionTrainUnet.h5'
            
        #model = load_model(str(MODEL_CHECKPOINTS/'UnetSmall_v4.h5'), custom_objects={'mean_iou': mean_iou, 'dice_coef': dice_coef, 'weighted_binary_crossentropy': WeightedLoss})    

        checkpointer = ModelCheckpoint(str(MODEL_CHECKPOINTS/backbone_model_name), verbose=1, save_best_only=True)
        earlystopping = EarlyStopping(monitor='val_loss', patience=30)
        schedule = lrate.SGDRScheduler(min_lr=1e-4,
                                 max_lr=1e-3,
                                 steps_per_epoch=np.ceil(train_number/BATCH_SIZE),
                                 lr_decay=0.8,
                                 cycle_length=15,
                                 mult_factor=1.25)

        ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=3, verbose=1, 
                                 mode='auto', cooldown=0, min_lr=9e-5)
        if scheduler == 'Adam':          
            callbacks = [checkpointer, earlystopping, ReduceLR]
        elif scheduler == 'SGD':
            callbacks = [checkpointer, earlystopping, schedule]
            
        if map_weights == 'no':
        
            WeightedLoss = create_weighted_binary_crossentropy(1, 1.5)  
            
        elif map_weights == 'yes':

            WeightedLoss = create_weighted_binary_crossentropy_2(1.05, 1)   

        if compiler == 'Adam':          
            Adam = optimizers.Adam(lr=0.002)
            model.compile(optimizer=Adam, loss=WeightedLoss, metrics=[mean_iou, dice_coef])
        elif compiler == 'SGD':
            SGD = optimizers.SGD(lr=0.001, momentum = 0.9)
            model.compile(optimizer=SGD, loss=WeightedLoss, metrics=[mean_iou, dice_coef])
            
        results = model.fit_generator(train_generator, 
                                      steps_per_epoch=train_number/BATCH_SIZE,
                                      validation_data=valid_generator, 
                                      validation_steps=valid_number/VALID_BATCH_SIZE, 
                                      callbacks=callbacks,
                                      epochs=5) 

    ############TRAIN ALL#####################################
    
    if map_weights == 'no':
        
        WeightedLoss = create_weighted_binary_crossentropy(1, 1.5)  
            
    elif map_weights == 'yes':

        WeightedLoss = create_weighted_binary_crossentropy_2(1.05, 1)  
        
    fine_model = load_model(str(MODEL_CHECKPOINTS/backbone_model_name), custom_objects={'mean_iou': mean_iou, 'dice_coef': dice_coef, 'weighted_binary_crossentropy': WeightedLoss})    
    # set_trainable(model)

    for layer in fine_model.layers[:263]:
        layer.trainable = False
    for layer in fine_model.layers[263:]:
        layer.trainable = True
        
    # print(K.eval(model.optimizer.lr))
    # print(K.get_value(model.optimizer.lr))
    # To set learning rate
    # K.set_value(model.optimizer.lr, 0.0006)
    # print(K.get_value(model.optimizer.lr)) 
        
    checkpointer = ModelCheckpoint(str(MODEL_CHECKPOINTS/fine_model_name), verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=30)

    schedule = lrate.SGDRScheduler(min_lr=1e-4,
                             max_lr=1e-3,
                             steps_per_epoch=np.ceil(train_number/BATCH_SIZE),
                             lr_decay=0.8,
                             cycle_length=15,
                             mult_factor=1.25)

    ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=3, verbose=1, 
                             mode='auto', cooldown=0, min_lr=9e-5)
    if scheduler == 'Adam':          
        callbacks = [checkpointer, earlystopping, ReduceLR]
    elif scheduler == 'SGD':
        callbacks = [checkpointer, earlystopping, schedule]
              
        
    if compiler == 'Adam':          
        Adam = optimizers.Adam(lr=0.0006)
        fine_model.compile(optimizer=Adam, loss=WeightedLoss, metrics=[mean_iou, dice_coef])
    elif compiler == 'SGD':
        SGD = optimizers.SGD(lr=0.001, momentum = 0.9)
        model.compile(optimizer=SGD, loss=WeightedLoss, metrics=[mean_iou, dice_coef])
        
    results = fine_model.fit_generator(train_generator, 
                                  steps_per_epoch=train_number/BATCH_SIZE,
                                  validation_data=valid_generator, 
                                  validation_steps=valid_number/VALID_BATCH_SIZE, 
                                  callbacks=callbacks,
                                  epochs=200)                                


if __name__ == "__main__":

    train_generator, valid_generator = imageGenerator(color_mode='grayscale')

    FineTuning(train_generator,valid_generator,backbone_name='inceptionv3', scheduler = 'Adam', compiler='Adam',
       map_weights = 'no', backbone_model_name = 'FreezeInceptionTrainUnet.h5', fine_model_name = 'TrainIncTrainUnet.h5', full_train = False)
                      
                      
                      
