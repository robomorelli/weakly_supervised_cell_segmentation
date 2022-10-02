#!/usr/bin/env python
# coding: utf-8

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
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import initializers, layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
from keras import optimizers
import keras.applications as ka
from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.utils import set_trainable

from config_script import *

tot_img_after_aug = count_files_in_directory(0, dir_list = ALL_IMAGES)

BATCH_SIZE = 2
VALID_BATCH_SIZE = 2
seed1=1 
val_percentage = 0.20
valid_number = int(tot_img_after_aug*val_percentage)
train_number = tot_img_after_aug - valid_number

IMG_CHANNELS = 3
IMG_HEIGHT = 400
IMG_WIDTH = 400

smooth = 1.

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

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def imageGenerator():

    image_datagen = ImageDataGenerator(rescale=1./255, validation_split = val_percentage)
    
    mask_datagen = ImageDataGenerator(rescale=1./255, validation_split = val_percentage, dtype=bool)
  
    
    train_image_generator = image_datagen.flow_from_directory(PRED_IMAGES.parent,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=BATCH_SIZE,
                                                              class_mode = None, seed = seed1, subset = 'training',color_mode = 'grayscale')
    train_mask_generator = mask_datagen.flow_from_directory(ALL_MASKS.parent,
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=BATCH_SIZE,
                                                            class_mode = None, seed = seed1, subset = 'training',
                                                            color_mode = 'grayscale')
    
    valid_image_generator = image_datagen.flow_from_directory(PRED_IMAGES.parent,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=VALID_BATCH_SIZE,
                                                              class_mode = None, seed = seed1, subset = 'validation',color_mode = 'grayscale')
    valid_mask_generator = mask_datagen.flow_from_directory(ALL_MASKS.parent,
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=VALID_BATCH_SIZE,
                                                            class_mode = None, seed = seed1, subset = 'validation',
                                                            color_mode = 'grayscale')
    
    train_generator = zip(train_image_generator,train_mask_generator)
    valid_generator = zip(valid_image_generator,valid_mask_generator)
    return train_generator, valid_generator

def unetRBG_postTrain(train_generator, valid_generator):

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, 1))
    
    #c0 = Conv2D(1,(1,1),activation='elu',kernel_initializer='he_normal',padding='same') (inputs)
    
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    
    #c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    #c2 = Dropout(0.1) (c2)
    #c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    #p2 = MaxPooling2D((2, 2)) (c2)
    
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    
    #c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    #c4 = Dropout(0.2) (c4)
    #c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    #p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)
    
    #u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    #u6 = concatenate([u6, c4])
    #c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    #c6 = Dropout(0.2) (c6)
    #c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)
    
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)
    
    #u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    #u8 = concatenate([u8, c2])
    #c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    #c8 = Dropout(0.1) (c8)
    #c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)
    
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    
    Adam = optimizers.Adam(lr=0.01)  
    model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=[mean_iou])
    
    checkpointer = ModelCheckpoint(str(MODEL_CHECKPOINTS/'UnetRGB_postTrain.h5'), verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=15)
    callbacks = [checkpointer, earlystopping]
    
    #model = load_model(str(MODEL_CHECKPOINTS/'UnetRGB_crop_v3.h5'), custom_objects={'mean_iou': mean_iou, 'dice_coef':dice_coef})
    
    results = model.fit_generator(train_generator, 
                                  steps_per_epoch=train_number/BATCH_SIZE,
                                  validation_data=valid_generator, 
                                  validation_steps=valid_number/VALID_BATCH_SIZE, 
                                  callbacks=callbacks,
                                  epochs=50)
        
train_generator, valid_generator = imageGenerator()
unetRBG_postTrain(train_generator,valid_generator)
