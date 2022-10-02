#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
import keras
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Conv2D, UpSampling2D, Lambda
from keras.models import Model, load_model
from keras.layers import Input
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

BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
seed1=1 
val_percentage = 0.20
valid_number = int(tot_img_after_aug*val_percentage)
train_number = tot_img_after_aug - valid_number

IMG_CHANNELS = 3
IMG_HEIGHT = 512
IMG_WIDTH = 512

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

def create_weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true*one_weight + (1. - y_true)*zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy

def rgb2hsv(image):

    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(image)
    image = cv2.merge([h*1./358, s*1./255, v*1./255])
    return image

def imageGenerator():

    image_datagen = ImageDataGenerator(rescale=1./255,validation_split = val_percentage)
                                       #preprocessing_function=rgb2hsv)
    mask_datagen = ImageDataGenerator(rescale=1./255, validation_split = val_percentage, dtype=bool)
    
    train_image_generator = image_datagen.flow_from_directory(ALL_IMAGES.parent,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=BATCH_SIZE,
                                                              class_mode = None, seed = seed1, subset = 'training')
    train_mask_generator = mask_datagen.flow_from_directory(ALL_MASKS.parent,
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=BATCH_SIZE,
                                                            class_mode = None, seed = seed1, subset = 'training',
                                                            color_mode = 'grayscale')
    
    valid_image_generator = image_datagen.flow_from_directory(ALL_IMAGES.parent,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=VALID_BATCH_SIZE,
                                                              class_mode = None, seed = seed1, subset = 'validation')
    valid_mask_generator = mask_datagen.flow_from_directory(ALL_MASKS.parent,
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),batch_size=VALID_BATCH_SIZE,
                                                            class_mode = None, seed = seed1, subset = 'validation',
                                                            color_mode = 'grayscale')
    
    train_generator = zip(train_image_generator,train_mask_generator)
    valid_generator = zip(valid_image_generator,valid_mask_generator)
    return train_generator, valid_generator


def unetVGG(train_generator, valid_generator):

    model = Unet(input_shape=(None, None, IMG_CHANNELS), backbone_name='vgg16', encoder_weights='imagenet', encoder_freeze=True)
    Adam = optimizers.Adam(lr=0.001)  
    model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=[mean_iou, dice_coef])
    
    model = load_model(str(MODEL_CHECKPOINTS/'freezedVggTrainedUnet.h5'), custom_objects={'mean_iou': mean_iou, 'dice_coef': dice_coef})
    
    checkpointer = ModelCheckpoint(str(MODEL_CHECKPOINTS/'freezedVggTrainedUnet.h5'), verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=10)      
    callbacks = [checkpointer, earlystopping]
    
    #model.fit_generator(
        #train_generator,
        #steps_per_epoch=train_number/BATCH_SIZE,
        #validation_data = valid_generator,
        #validation_steps=valid_number/VALID_BATCH_SIZE,
        #callbacks = callbacks,
        #epochs=2)
    
    set_trainable(model) 
    
    checkpointer = ModelCheckpoint(str(MODEL_CHECKPOINTS/'TrainedVggTrainedUnet_crop.h5'), verbose=1, save_best_only=True) 
    earlystopping = EarlyStopping(monitor='val_loss', patience=15)   
    tensorboard = tb.TrainValTensorBoard(log_dir=str(RESULTS_DIRECTORY / 'logs'), write_graph=False, write_images=True,
                                batch_size = BATCH_SIZE, write_grads=True)
    plot_losses = plots.TrainingPlot()
    schedule = lrate.SGDRScheduler(min_lr=1e-10,
                             max_lr=1e-2,
                             steps_per_epoch=np.ceil(train_number/BATCH_SIZE),
                             lr_decay=0.9,
                             cycle_length=30,
                             mult_factor=1.5)  
    callbacks = [checkpointer, earlystopping, tensorboard, plot_losses, schedule]
    
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_number/BATCH_SIZE,
        validation_data = valid_generator,
        validation_steps=valid_number/VALID_BATCH_SIZE,
        callbacks = callbacks,
        epochs=100)
    
def unetSmall(train_generator, valid_generator):
    
    # Build U-Net model
    inputs = Input((None, None, 3))
    
    c1 = Conv2D(8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
    c1 = BatchNormalization()(c1) 
    c1 = Dropout(0.1) (c1)           
    c1 = Conv2D(8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    
    c2 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    
    c3 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    
    c4 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    
    c5 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)
    c5 = BatchNormalization()(c5)
    
    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = BatchNormalization()(u6)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)
    c6 = BatchNormalization()(c6)
    
    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = BatchNormalization()(u7)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)
    c7 = BatchNormalization()(c7)
    
    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = BatchNormalization()(u8)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)
    c8 = BatchNormalization()(c8)
    
    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = BatchNormalization()(u9)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)
    c9 = BatchNormalization()(c9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary() 
    
    WeightedLoss = create_weighted_binary_crossentropy(1, 2)    
    
    Adam = optimizers.Adam(lr=0.001)  
    model.compile(optimizer=Adam, loss=WeightedLoss, metrics=[mean_iou, dice_coef])
    
    #model = load_model(str(MODEL_CHECKPOINTS/'UnetSmall_v4.h5'), custom_objects={'mean_iou': mean_iou, 'dice_coef': dice_coef, 'weighted_binary_crossentropy': WeightedLoss})    
    
    checkpointer = ModelCheckpoint(str(MODEL_CHECKPOINTS/'UnetSmallYellow.h5'), verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=30)
    tensorboard = tb.TrainValTensorBoard(log_dir=str(RESULTS_DIRECTORY / 'logs'), write_graph=True, write_images=True,
                                         batch_size = BATCH_SIZE, write_grads=False) 
    #tensorboard = TensorBoard(log_dir=str(RESULTS_DIRECTORY / 'logs'), histogram_freq=1, batch_size=BATCH_SIZE, write_graph=True, 
                              #write_grads=False, write_images=True, embeddings_freq=1, embeddings_layer_names=None, 
                              #embeddings_metadata=None, embeddings_data=None, update_freq='batch')
    plot_losses = plots.TrainingPlot()
    # schedule = lrate.SGDRScheduler(min_lr=1e-5,
                             # max_lr=1e-2,
                             # steps_per_epoch=np.ceil(train_number/BATCH_SIZE),
                             # lr_decay=0.9,
                             # cycle_length=20,
                             # mult_factor=1.5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=3, verbose=1)							 
    callbacks = [checkpointer, earlystopping, tensorboard, reduce_lr]
    
    results = model.fit_generator(train_generator, 
                                  steps_per_epoch=train_number/BATCH_SIZE,
                                  validation_data=valid_generator, 
                                  validation_steps=valid_number/VALID_BATCH_SIZE, 
                                  callbacks=callbacks,
                                  epochs=200)
        
def unetRGB(train_generator, valid_generator):
    
    inputs = Input((None, None, IMG_CHANNELS))
    
    #c0 = Conv2D(1,(1,1),activation='elu',kernel_initializer='he_normal',padding='same') (inputs)
    
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)
    c5 = BatchNormalization()(c5)
    
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)
    c6 = BatchNormalization()(c6)
    
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)
    c7 = BatchNormalization()(c7)
    
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)
    c8 = BatchNormalization()(c8)
    
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)
    c9 = BatchNormalization()(c9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    
    Adam = optimizers.Adam(lr=0.001)  
    model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=[mean_iou, dice_coef])
    
    checkpointer = ModelCheckpoint(str(MODEL_CHECKPOINTS/'UnetRGB_BigTrain_v3.h5'), verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=50)
 #   tensorboard = TensorBoard(log_dir=str(RESULTS_DIRECTORY/'logs'), histogram_freq=0, write_graph=True, write_images=False)        
 #   plot_losses = plots.TrainingPlot()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=2, verbose=1)
 #   schedule = lrate.SGDRScheduler(min_lr=1e-6,
 #                            max_lr=1e-3,
 #                            steps_per_epoch=np.ceil(train_number/BATCH_SIZE),
 #                            lr_decay=0.9,
 #                            cycle_length=30,
 #                            mult_factor=1.5)    
    callbacks = [checkpointer, earlystopping, reduce_lr]
    
    model = load_model(str(MODEL_CHECKPOINTS/'UnetRGB_BigTrain_v2.h5'), custom_objects={'mean_iou': mean_iou, 'dice_coef': dice_coef})
    
    results = model.fit_generator(train_generator, 
                                  steps_per_epoch=train_number/BATCH_SIZE,
                                  validation_data=valid_generator, 
                                  validation_steps=valid_number/VALID_BATCH_SIZE, 
                                  callbacks=callbacks,
                                  epochs=2000)
    

def unetGRAY(train_generator, valid_generator):
    
    inputs = Input((None, None, 1))
        
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)
    c5 = BatchNormalization()(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)
    c6 = BatchNormalization()(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)
    c7 = BatchNormalization()(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)
    c8 = BatchNormalization()(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)
    c9 = BatchNormalization()(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    Adam = optimizers.Adam(lr=0.001)  
    model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=[mean_iou, dice_coef])
    
    checkpointer = ModelCheckpoint(str(MODEL_CHECKPOINTS/'UnetGRAY_Aug_v2.h5'), verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=30)
    tensorboard = TensorBoard(log_dir=str(RESULTS_DIRECTORY/'logs'), histogram_freq=0, write_graph=True, write_images=False)        
    plot_losses = plots.TrainingPlot()
    schedule = lrate.SGDRScheduler(min_lr=1e-10,
                             max_lr=1e-2,
                             steps_per_epoch=np.ceil(train_number/BATCH_SIZE),
                             lr_decay=0.9,
                             cycle_length=5,
                             mult_factor=1.5)       
    callbacks = [checkpointer, earlystopping, plot_losses, schedule]
    callbacks_tb = [checkpointer, earlystopping, tensorboard, schedule]
    
    #model = load_model(str(MODEL_CHECKPOINTS/'UnetRGB.h5'), custom_objects={'mean_iou': mean_iou})
    model.load_weights(str(MODEL_CHECKPOINTS/'UnetGRAY_Aug.h5'))
    model.summary()
    
    model.fit_generator(train_generator, 
                        steps_per_epoch=train_number/BATCH_SIZE,
                        validation_data=valid_generator, 
                        validation_steps=valid_number/VALID_BATCH_SIZE, 
                        callbacks=callbacks,
                        epochs=100)

def unetLUCA(train_generator, valid_generator):
    
	inputs = Input((None, None, 1))

	c1 = Conv2D(16, (3, 3), activation='relu',
		    kernel_initializer='he_normal', padding='same')(inputs)
	c1 = Dropout(0.15)(c1)
	c1 = Conv2D(16, (3, 3), activation='relu',
		    kernel_initializer='he_normal', padding='same')(c1)
	p1 = MaxPooling2D((2, 2))(c1)

	c2 = Conv2D(32, (3, 3), activation='relu',
		    kernel_initializer='he_normal', padding='same')(p1)
	c2 = Dropout(0.25)(c2)
	c2 = Conv2D(32, (3, 3), activation='relu',
		    kernel_initializer='he_normal', padding='same')(c2)

	u3 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c2)
	u3 = concatenate([u3, c1], axis=3)
	c3 = Conv2D(16, (3, 3), activation='relu',
		    kernel_initializer='he_normal', padding='same')(u3)
	c3 = Dropout(0.15)(c3)
	c3 = Conv2D(16, (3, 3), activation='relu',
		    kernel_initializer='he_normal', padding='same')(c3)

	outputs = Conv2D(1, (1, 1), activation='sigmoid')(c3)

	model = Model(inputs=[inputs], outputs=[outputs])
	Adam = optimizers.Adam(lr=0.001)  
	model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=[mean_iou, dice_coef])
	
	model.fit_generator(train_generator, 
	                    steps_per_epoch=train_number/BATCH_SIZE,
	                    validation_data=valid_generator, 
	                    validation_steps=valid_number/VALID_BATCH_SIZE, 
	                    epochs=100)	
if __name__ == "__main__":
    train_generator, valid_generator = imageGenerator()
    unetSmall(train_generator,valid_generator)
