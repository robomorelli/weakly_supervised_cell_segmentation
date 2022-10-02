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
from keras.utils import multi_gpu_model


from importlib import import_module
plots = import_module('051_plotscalars')
tb = import_module('050_tensorboard')
lrate = import_module('043_learning_rate')

from config_script import *

#tot_img_after_aug = count_files_in_directory(0, dir_list = ALL_IMAGES)
tot_img_after_aug = 18001

BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
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
        
        weights *= 6.538698

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
                                  
def ResUnet(train_generator, valid_generator, scheduler = 'Adam'):
    
    inputs = Input((None, None, 3))

    X_shortcut = Conv2D(8, (1, 1), padding='same',  kernel_initializer='he_normal')(inputs)
    
    c1 = Conv2D(8, (3, 3), padding='same',  kernel_initializer='he_normal')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation('elu')(c1)

    c1 = Conv2D(8, (3, 3), padding='same', kernel_initializer='he_normal')(c1)

    c1 = Add()([c1, X_shortcut])

    p1 = MaxPooling2D((2, 2))(c1)

    ##########################################
    X_shortcut = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_normal')(p1)

    c2 = BatchNormalization()(p1)
    c2 = Activation('elu')(c2)
    c2 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(c2)

    c2 = BatchNormalization()(c2)
    c2 = Activation('elu')(c2)
    c2 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(c2)

    c2 = Add()([c2, X_shortcut])

    p2 = MaxPooling2D((2, 2))(c2)

    ##########################################
    X_shortcut = Conv2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(p2)
    
    c3 = BatchNormalization()(p2)
    c3 = Activation('elu')(c3)
    c3 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(c3)

    c3 = BatchNormalization()(c3)
    c3 = Activation('elu')(c3)
    c3 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(c3)

    c3 = Add()([c3, X_shortcut])

    p3 = MaxPooling2D((2, 2))(c3)

    ###################Bridge#######################
    X_shortcut = Conv2D(64, (1, 1), padding='same',kernel_initializer='he_normal')(p3)
    
    c4 = BatchNormalization()(p3)
    c4 = Activation('elu')(c4)
    c4 = Conv2D(64, (3, 3), padding='same',kernel_initializer='he_normal')(c4)

    c4 = BatchNormalization()(c4)
    c4 = Activation('elu')(c4)
    c4 = Conv2D(64, (3, 3), padding='same',kernel_initializer='he_normal')(c4)

    c4 = Add()([c4, X_shortcut])

    X_shortcut = c4

    c5 = BatchNormalization()(c4)
    c5 = Activation('elu')(c5)
    c5 = Conv2D(64, (3, 3), padding='same',kernel_initializer='he_normal')(c5)

    c5 = BatchNormalization()(c5)
    c5 = Activation('elu')(c5)
    c5 = Conv2D(64, (3, 3), padding='same',kernel_initializer='he_normal')(c5)

    c5 = Add()([c5, X_shortcut])


    ###################END BRIDGE#######################


    u6 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c3])

    X_shortcut = Conv2D(32, (1, 1), padding='same', kernel_initializer='he_normal') (u6)

    c6 = BatchNormalization()(u6)
    c6 = Activation('elu')(c6)
    c6 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(c6)

    c6 = BatchNormalization()(c6)
    c6 = Activation('elu')(c6)
    c6 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(c6)

    c6 = Add()([c6, X_shortcut])

    ################################################

    u7 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c2])

    X_shortcut = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_normal') (u7)

    c7 = BatchNormalization()(u7)
    c7 = Activation('elu')(c7)
    c7 = Conv2D(16, (3, 3), padding='same',kernel_initializer='he_normal')(c7)

    c7 = BatchNormalization()(c7)
    c7 = Activation('elu')(c7)
    c7 = Conv2D(16, (3, 3), padding='same',kernel_initializer='he_normal')(c7)

    c7 = Add()([c7, X_shortcut])

    u8 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c1])

    X_shortcut = Conv2D(8, (1, 1), padding='same', kernel_initializer='he_normal') (u8)

    c8 = BatchNormalization()(u8)
    c8 = Activation('elu')(c8)
    c8 = Conv2D(8, (3, 3), padding='same',kernel_initializer='he_normal')(c8)

    c8 = BatchNormalization()(c8)
    c8 = Activation('elu')(c8)
    c8 = Conv2D(8, (3, 3), padding='same',kernel_initializer='he_normal')(c8)

    c8 = Add()([c8, X_shortcut])

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c8)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    model.summary() 
    
    WeightedLoss = create_weighted_binary_crossentropy(1, 1.5)    
    
    Adam = optimizers.Adam(lr=0.0008)  
    model.compile(optimizer=Adam, loss=WeightedLoss, metrics=[mean_iou, dice_coef])
    
    model = load_model(str(MODEL_CHECKPOINTS/'ResNet_Small_v1.h4'), custom_objects={'mean_iou' : mean_iou, 'dice_coeff': dice_coeff, 'weighted_binary_crossentropy': WeightedLoss})
    
    #model = load_model(str(MODEL_CHECKPOINTS/'UnetSmall_v4.h5'), custom_objects={'mean_iou': mean_iou, 'dice_coef': dice_coef, 'weighted_binary_crossentropy': WeightedLoss})    
    
    checkpointer = ModelCheckpoint(str(MODEL_CHECKPOINTS/'ResUnetBasicSGD.h5'), verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=30)
    tensorboard = tb.TrainValTensorBoard(log_dir=str(RESULTS_DIRECTORY / 'logs'), write_graph=True, write_images=True,
                                         batch_size = BATCH_SIZE, write_grads=False) 
    #tensorboard = TensorBoard(log_dir=str(RESULTS_DIRECTORY / 'logs'), histogram_freq=1, batch_size=BATCH_SIZE, write_graph=True, 
                              #write_grads=False, write_images=True, embeddings_freq=1, embeddings_layer_names=None, 
                              #embeddings_metadata=None, embeddings_data=None, update_freq='batch')
    plot_losses = plots.TrainingPlot()
    schedule = lrate.SGDRScheduler(min_lr=1e-6,
                             max_lr=3e-4,
                             steps_per_epoch=np.ceil(train_number/BATCH_SIZE),
                             lr_decay=0.9,
                             cycle_length=20,
                             mult_factor=1.5)

    ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=3, verbose=1, 
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
                                  

def ResUnetIdentity(train_generator, valid_generator, scheduler = 'Adam',compiler = 'Adam',  n = 1, map_weights = 'no', model_name = 'ResUnetIdentity.h5'):
    
    inputs = Input((None, None, 3))

    #X_shortcut = Conv2D(4*n, (1, 1), padding='same',  kernel_initializer='he_normal')(inputs)
    #X_shortcut = BatchNormalization()(X_shortcut)

    c1 = Conv2D(4*n, (7, 7), padding='same',  kernel_initializer='he_normal')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation('elu')(c1)

    c1 = Conv2D(4*n, (3, 3), padding='same', kernel_initializer='he_normal')(c1)

    #c1 = Add()([c1, X_shortcut])

    p1 = MaxPooling2D((2, 2))(c1)

    ##########################################
    X_shortcut = Conv2D(8*n, (1, 1), padding='same', kernel_initializer='he_normal')(p1)
    X_shortcut = BatchNormalization()(X_shortcut)

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
    X_shortcut = BatchNormalization()(X_shortcut)
    
    c3 = BatchNormalization()(p2)
    c3 = Activation('elu')(c3)
    c3 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c3)

    c3 = BatchNormalization()(c3)
    c3 = Activation('elu')(c3)
    c3 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c3)

    c3 = Add()([c3, X_shortcut])

    p3 = MaxPooling2D((2, 2))(c3)

    ###################Bridge#######################
    X_shortcut = Conv2D(32*n, (1, 1), padding='same',kernel_initializer='he_normal')(p3)
    X_shortcut = BatchNormalization()(X_shortcut)
    
    c4 = BatchNormalization()(p3)
    c4 = Activation('elu')(c4)
    c4 = Conv2D(32*n, (5, 5), padding='same',kernel_initializer='he_normal')(c4)

    c4 = BatchNormalization()(c4)
    c4 = Activation('elu')(c4)
    c4 = Conv2D(32*n, (5, 5), padding='same',kernel_initializer='he_normal')(c4)

    #c4 = Add()([c4, X_shortcut])

    #X_shortcut = c4

    c5 = BatchNormalization()(c4)
    c5 = Activation('elu')(c5)
    c5 = Conv2D(32*n, (5, 5), padding='same',kernel_initializer='he_normal')(c5)

    c5 = BatchNormalization()(c5)
    c5 = Activation('elu')(c5)
    c5 = Conv2D(32*n, (5, 5), padding='same',kernel_initializer='he_normal')(c5)

    c5 = Add()([c5, X_shortcut])


    ###################END BRIDGE#######################


    u6 = Conv2DTranspose(16*n, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c3])

    X_shortcut  = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal') (u6)
    X_shortcut = BatchNormalization()(X_shortcut)

    c6 = BatchNormalization()(u6)
    c6 = Activation('elu')(c6)
    c6 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c6)

    c6 = BatchNormalization()(c6)
    c6 = Activation('elu')(c6)
    c6 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c6)

    c6 = Add()([c6, X_shortcut])

    ################################################

    u7 = Conv2DTranspose(8*n, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c2])
   
    X_shortcut = Conv2D(8*n, (1, 1), padding='same', kernel_initializer='he_normal') (u7)
    X_shortcut = BatchNormalization()(X_shortcut)

    c7 = BatchNormalization()(u7)
    c7 = Activation('elu')(c7)
    c7 = Conv2D(8*n, (3, 3), padding='same',kernel_initializer='he_normal')(c7)

    c7 = BatchNormalization()(c7)
    c7 = Activation('elu')(c7)
    c7 = Conv2D(8*n, (3, 3), padding='same',kernel_initializer='he_normal')(c7)

    c7 = Add()([c7, X_shortcut])

    u8 = Conv2DTranspose(4*n, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c1])

    X_shortcut = Conv2D(4*n, (1, 1), padding='same', kernel_initializer='he_normal') (u8)
    X_shortcut = BatchNormalization()(X_shortcut)

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

        WeightedLoss = create_weighted_binary_crossentropy_2(1.1, 1)     

    #model = multi_gpu_model(model, gpus=2)    

    if compiler == 'Adam':
       Adam = optimizers.Adam(lr=0.003)
       model.compile(optimizer=Adam, loss=WeightedLoss, metrics=[mean_iou, dice_coef])
    
    elif compiler == 'SGD':
       SGD = optimizers.SGD(lr=0.003)
       model.compile(optimizer=SGD, loss=WeightedLoss, metrics=[mean_iou, dice_coef]) 
   
    #model = load_model(str(MODEL_CHECKPOINTS/'ResUnetCnafCore1N42w0_11SGD.h5'), 
    custom_objects={'mean_iou': mean_iou, 'dice_coef': dice_coef, 'weighted_binary_crossentropy': WeightedLoss})    
    
    checkpointer = ModelCheckpoint(str(MODEL_CHECKPOINTS/model_name), verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=30)
    tensorboard = tb.TrainValTensorBoard(log_dir=str(RESULTS_DIRECTORY / 'logs'), write_graph=True, write_images=True,
                                         batch_size = BATCH_SIZE, write_grads=False) 
    #tensorboard = TensorBoard(log_dir=str(RESULTS_DIRECTORY / 'logs'), histogram_freq=1, batch_size=BATCH_SIZE, write_graph=True, 
                              #write_grads=False, write_images=True, embeddings_freq=1, embeddings_layer_names=None, 
                              #embeddings_metadata=None, embeddings_data=None, update_freq='batch')
    plot_losses = plots.TrainingPlot()
    schedule = lrate.SGDRScheduler(min_lr=1e-4,
                             max_lr=3e-3,
                             steps_per_epoch=np.ceil(train_number/BATCH_SIZE),
                             lr_decay=0.8,
                             cycle_length=10,
                             mult_factor=1)

    ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3, verbose=1, 
                             mode='auto', cooldown=0, min_lr=1e-5)
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
                                  
                                  
                                  

                                 
def DeepResUnet2MP(train_generator, valid_generator, scheduler = 'Adam', n = 1, map_weights = 'no', model_name = 'ResUnet2MP.h5'):
    
    n=n
    
    inputs = Input((None, None, 3))
    X_shortcut = Conv2D(8*n, (1, 1), padding='same',  kernel_initializer='he_normal')(inputs)

    c1 = Conv2D(8*n, (3, 3), padding='same',  kernel_initializer='he_normal')(X_shortcut)
    c1 = BatchNormalization()(c1)
    c1 = Activation('elu')(c1)

    c1 = Conv2D(8*n, (3, 3), padding='same', kernel_initializer='he_normal')(c1)

    c1 = Add()([c1, X_shortcut])

    # pc1 = ZeroPadding2D((1,1))(c1)
    pc1 = MaxPooling2D((2,2))(c1)

    ##########################################
    X_shortcut = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal')(pc1)

    c2 = BatchNormalization()(X_shortcut)
    c2 = Activation('elu')(c2)
    c2 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c2)

    c2 = BatchNormalization()(c2)
    c2 = Activation('elu')(c2)
    c2 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c2)

    c2 = Add()([c2, X_shortcut])

    # pc2 = ZeroPadding2D((1,1))(c2)
    pc2 = MaxPooling2D((2,2))(c2)

    ##########################################
    X_shortcut = Conv2D(32*n, (1, 1), padding='same', kernel_initializer='he_normal')(pc2)

    c3 = BatchNormalization()(X_shortcut)
    c3 = Activation('elu')(c3)
    c3 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer='he_normal')(c3)

    c3 = BatchNormalization()(c3)
    c3 = Activation('elu')(c3)
    c3 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer='he_normal')(c3)

    c3 = Add()([c3, X_shortcut])

    pc3 = ZeroPadding2D((1,1))(c3)
    pc3 = Conv2D(32*n, (3, 3),strides=(2,2), padding='valid', kernel_initializer='he_normal')(pc3)

    ##############################################
    X_shortcut = Conv2D(64*n, (1, 1), padding='same', kernel_initializer='he_normal')(pc3)

    c51 = BatchNormalization()(X_shortcut)
    c51 = Activation('elu')(c51)
    c51 = Conv2D(16*n, (3, 3),dilation_rate = (6,6),padding='same', kernel_initializer='he_normal')(c51)

    c52 = BatchNormalization()(X_shortcut)
    c52 = Activation('elu')(c52)
    c52 = Conv2D(16*n, (3, 3), dilation_rate = (12,12), padding='same', kernel_initializer='he_normal')(c52)

    c53 = BatchNormalization()(X_shortcut)
    c53 = Activation('elu')(c53)
    c53 = Conv2D(16*n, (3, 3), dilation_rate = (18,18), padding='same', kernel_initializer='he_normal')(c53)

    c54 = BatchNormalization()(X_shortcut)
    c54 = Activation('elu')(c54)
    c54 = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal')(c54)

    c5 = concatenate([c51,c52,c53,c54])
    #c5 = SpatialDropout2D(0.2)(c5)

    c5 = Conv2D(64*n, (1, 1), padding='same', kernel_initializer='he_normal')(c5)

    c5 = Add()([c5, X_shortcut])
    #################################################################################

    u6 = Conv2DTranspose(32*n, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c3])
    u6 = Conv2D(32*n, (1, 1), padding='same', kernel_initializer='he_normal') (u6)

    X_shortcut = u6

    c6 = BatchNormalization()(X_shortcut)
    c6 = Activation('elu')(c6)
    c6 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer='he_normal')(c6)

    c6 = BatchNormalization()(c6)
    c6 = Activation('elu')(c6)
    c6 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer='he_normal')(c6)

    c6 = Add()([c6, X_shortcut])

    ################################################

    u7 = Conv2DTranspose(16*n, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c2])
    u7 = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal') (u7)

    X_shortcut = u7

    c7 = BatchNormalization()(X_shortcut)
    c7 = Activation('elu')(c7)
    c7 = Conv2D(16*n, (3, 3), padding='same',kernel_initializer='he_normal')(c7)

    c7 = BatchNormalization()(c7)
    c7 = Activation('elu')(c7)
    c7 = Conv2D(16*n, (3, 3), padding='same',kernel_initializer='he_normal')(c7)

    c7 = Add()([c7, X_shortcut])

    #####################################################################################

    u8 = Conv2DTranspose(8*n, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c1])
    u8 = Conv2D(8*n, (1, 1), padding='same', kernel_initializer='he_normal') (u8)

    X_shortcut = u8

    c8 = BatchNormalization()(X_shortcut)
    c8 = Activation('elu')(c8)
    c8 = Conv2D(8*n, (3, 3), padding='same',kernel_initializer='he_normal')(c8)      

    c8 = BatchNormalization()(c8)
    c8 = Activation('elu')(c8)
    c8 = Conv2D(8*n, (3, 3), padding='same',kernel_initializer='he_normal')(c8)

    c8 = Add()([c8, X_shortcut])

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c8)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    model.summary()

    if map_weights == 'no':
    
        WeightedLoss = create_weighted_binary_crossentropy(1, 1.5)  
        
    elif map_weights == 'yes':

        WeightedLoss = create_weighted_binary_crossentropy_2(1, 1.3)     
    
    Adam = optimizers.Adam(lr=0.001)
    #model = multi_gpu_model(model, gpus=2)
  
    model.compile(optimizer=Adam, loss=WeightedLoss, metrics=[mean_iou, dice_coef])
   
    #model = load_model(str(MODEL_CHECKPOINTS/'UnetSmall_v4.h5'), custom_objects={'mean_iou': mean_iou, 'dice_coef': dice_coef, 'weighted_binary_crossentropy': WeightedLoss})    
    
    checkpointer = ModelCheckpoint(str(MODEL_CHECKPOINTS/model_name), verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=30)
    tensorboard = tb.TrainValTensorBoard(log_dir=str(RESULTS_DIRECTORY / 'logs'), write_graph=True, write_images=True,
                                         batch_size = BATCH_SIZE, write_grads=False) 
    #tensorboard = TensorBoard(log_dir=str(RESULTS_DIRECTORY / 'logs'), histogram_freq=1, batch_size=BATCH_SIZE, write_graph=True, 
                              #write_grads=False, write_images=True, embeddings_freq=1, embeddings_layer_names=None, 
                              #embeddings_metadata=None, embeddings_data=None, update_freq='batch')
    plot_losses = plots.TrainingPlot()
    schedule = lrate.SGDRScheduler(min_lr=1e-6,
                             max_lr=3e-4,
                             steps_per_epoch=np.ceil(train_number/BATCH_SIZE),
                             lr_decay=0.9,
                             cycle_length=20,
                             mult_factor=1.5)

    ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3, verbose=1, 
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
                                  
                                  


                                  
def ResUnetBig(train_generator, valid_generator, scheduler = 'Adam',n=1, map_weights = 'no', model_name = 'ResUnetBig.h5'):
    
    n = n
    inputs = Input((None, None, 3))

    X_shortcut = Conv2D(8, (1, 1), padding='same',  kernel_initializer='he_normal')(inputs)

    c1 = Conv2D(8, (3, 3), padding='same',  kernel_initializer='he_normal')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation('elu')(c1)

    c1 = Conv2D(8, (3, 3), padding='same', kernel_initializer='he_normal')(c1)

    c1 = Add()([c1, X_shortcut])

    p1 = MaxPooling2D((2, 2))(c1)

    ##########################################
    X_shortcut = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_normal')(p1)

    c2 = BatchNormalization()(p1)
    c2 = Activation('elu')(c2)
    c2 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(c2)

    c2 = BatchNormalization()(c2)
    c2 = Activation('elu')(c2)
    c2 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(c2)

    c2 = Add()([c2, X_shortcut])

    p2 = MaxPooling2D((2, 2))(c2)

    ##########################################
    X_shortcut = Conv2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(p2)

    c3 = BatchNormalization()(p2)
    c3 = Activation('elu')(c3)
    c3 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(c3)

    c3 = BatchNormalization()(c3)
    c3 = Activation('elu')(c3)
    c3 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(c3)

    c3 = Add()([c3, X_shortcut])

    p3 = MaxPooling2D((2, 2))(c3)

    ###################Bridge#######################
    X_shortcut = Conv2D(64, (1, 1), padding='same',kernel_initializer='he_normal')(p3)

    c4 = BatchNormalization()(p3)
    c4 = Activation('elu')(c4)
    c4 = Conv2D(64, (3, 3), padding='same',kernel_initializer='he_normal')(c4)

    c4 = BatchNormalization()(c4)
    c4 = Activation('elu')(c4)
    c4 = Conv2D(64, (3, 3), padding='same',kernel_initializer='he_normal')(c4)

    c4 = Add()([c4, X_shortcut])

    X_shortcut = c4

    c5 = BatchNormalization()(c4)
    c5 = Activation('elu')(c5)
    c5 = Conv2D(64, (3, 3), padding='same',kernel_initializer='he_normal')(c5)

    c5 = BatchNormalization()(c5)
    c5 = Activation('elu')(c5)
    c5 = Conv2D(64, (3, 3), padding='same',kernel_initializer='he_normal')(c5)

    c5 = Add()([c5, X_shortcut])
      
    X_shortcut = c5

    c6 = BatchNormalization()(c5)
    c6 = Activation('elu')(c6)
    c6 = Conv2D(64, (3, 3), padding='same',kernel_initializer='he_normal')(c6)

    c6 = BatchNormalization()(c6)
    c6 = Activation('elu')(c6)
    c6 = Conv2D(64, (3, 3), padding='same',kernel_initializer='he_normal')(c6)

    c6 = Add()([c6, X_shortcut])
    
    X_shortcut = c6

    c7 = BatchNormalization()(c6)
    c7 = Activation('elu')(c7)
    c7 = Conv2D(64, (3, 3), padding='same',kernel_initializer='he_normal')(c7)

    c7 = BatchNormalization()(c7)
    c7 = Activation('elu')(c7)
    c7 = Conv2D(64, (3, 3), padding='same',kernel_initializer='he_normal')(c7)

    c7 = Add()([c7, X_shortcut])


    ###################END BRIDGE#######################

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c3])
    X_shortcut = Conv2D(32, (1, 1), padding='same', kernel_initializer='he_normal') (u8)

    c8 = BatchNormalization()(u8)
    c8 = Activation('elu')(c8)
    c8 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(c8)

    c8 = BatchNormalization()(c8)
    c8 = Activation('elu')(c8)
    c8 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(c8)

    c8 = Add()([c8, X_shortcut])

    ################################################

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c2])
    X_shortcut = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_normal') (u9)

    c9 = BatchNormalization()(u9)
    c9 = Activation('elu')(c9)
    c9 = Conv2D(16, (3, 3), padding='same',kernel_initializer='he_normal')(c9)

    c9 = BatchNormalization()(c9)
    c9 = Activation('elu')(c9)
    c9 = Conv2D(16, (3, 3), padding='same',kernel_initializer='he_normal')(c9)

    c9 = Add()([c9, X_shortcut])
        
    u10 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c9)
    u10 = concatenate([u10, c1])
    X_shortcut = Conv2D(8, (1, 1), padding='same', kernel_initializer='he_normal') (u10)

    c10 = BatchNormalization()(u10)
    c10 = Activation('elu')(c10)
    c10 = Conv2D(8, (3, 3), padding='same',kernel_initializer='he_normal')(c10)

    c10 = BatchNormalization()(c10)
    c10 = Activation('elu')(c10)
    c10 = Conv2D(8, (3, 3), padding='same',kernel_initializer='he_normal')(c10)

    c10 = Add()([c10, X_shortcut])

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c10)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    model.summary() 
    
    if map_weights == 'no':
    
        WeightedLoss = create_weighted_binary_crossentropy(1, 1.5)  
        
    elif map_weights == 'yes':

        WeightedLoss = create_weighted_binary_crossentropy_2(1, 1.5)     
    
    Adam = optimizers.Adam(lr=0.0008)
    #model = multi_gpu_model(model, gpus=2)
  
    model.compile(optimizer=Adam, loss=WeightedLoss, metrics=[mean_iou, dice_coef])
    
    #model = load_model(str(MODEL_CHECKPOINTS/'UnetSmall_v4.h5'), custom_objects={'mean_iou': mean_iou, 'dice_coef': dice_coef, 'weighted_binary_crossentropy': WeightedLoss})    
    
    checkpointer = ModelCheckpoint(str(MODEL_CHECKPOINTS/model_name), verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=30)
    tensorboard = tb.TrainValTensorBoard(log_dir=str(RESULTS_DIRECTORY / 'logs'), write_graph=True, write_images=True,
                                         batch_size = BATCH_SIZE, write_grads=False) 
    #tensorboard = TensorBoard(log_dir=str(RESULTS_DIRECTORY / 'logs'), histogram_freq=1, batch_size=BATCH_SIZE, write_graph=True, 
                              #write_grads=False, write_images=True, embeddings_freq=1, embeddings_layer_names=None, 
                              #embeddings_metadata=None, embeddings_data=None, update_freq='batch')
    plot_losses = plots.TrainingPlot()
    
    # schedule = lrate.SGDRScheduler(min_lr=1e-6,
                             # max_lr=3e-4,
                             # steps_per_epoch=np.ceil(train_number/BATCH_SIZE),
                             # lr_decay=0.9,
                             # cycle_length=20,
                             # mult_factor=1.5)    
                             
    ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=3, verbose=1, 
                             mode='auto', cooldown=0, min_lr=3e-7)
                             
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
                                  

                                  
                                  
def unetSmall(train_generator, valid_generator, scheduler = 'Adam', n = 1, map_weights = 'no', model_name = 'ResUnetIdentity.h5'):
    
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
    
    if map_weights == 'no':
    
        WeightedLoss = create_weighted_binary_crossentropy(1, 1.5)  
        
    elif map_weights == 'yes':

        WeightedLoss = create_weighted_binary_crossentropy_2(1, 1.3)     
    
    Adam = optimizers.Adam(lr=0.0008)  
    model.compile(optimizer=Adam, loss=WeightedLoss, metrics=[mean_iou, dice_coef])
    
    #model = load_model(str(MODEL_CHECKPOINTS/'UnetSmall_v4.h5'), custom_objects={'mean_iou': mean_iou, 'dice_coef': dice_coef, 'weighted_binary_crossentropy': WeightedLoss})    
    
    checkpointer = ModelCheckpoint(str(MODEL_CHECKPOINTS/model_name), verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=30)
    tensorboard = tb.TrainValTensorBoard(log_dir=str(RESULTS_DIRECTORY / 'logs'), write_graph=True, write_images=True,
                                         batch_size = BATCH_SIZE, write_grads=False) 
    #tensorboard = TensorBoard(log_dir=str(RESULTS_DIRECTORY / 'logs'), histogram_freq=1, batch_size=BATCH_SIZE, write_graph=True, 
                              #write_grads=False, write_images=True, embeddings_freq=1, embeddings_layer_names=None, 
                              #embeddings_metadata=None, embeddings_data=None, update_freq='batch')
    plot_losses = plots.TrainingPlot()
    schedule = lrate.SGDRScheduler(min_lr=1e-6,
                             max_lr=3e-4,
                             steps_per_epoch=np.ceil(train_number/BATCH_SIZE),
                             lr_decay=0.9,
                             cycle_length=20,
                             mult_factor=1.5)

    ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3, verbose=1, 
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
                                  
                                  
                                  
                                  
def InceptionUnetBasic(train_generator,valid_generator, scheduler = 'Adam', n = 1, map_weights = 'yes', 
                   model_name = 'InceptionUnet.h5'):

    inputs = Input((None, None, 3))

    X_shortcut = Conv2D(16, (1, 1), padding='same',  kernel_initializer='he_normal')(inputs)

    c1 = Conv2D(16, (3, 3), padding='same',  kernel_initializer='he_normal')(X_shortcut)
    c1 = BatchNormalization()(c1)
    c1 = Activation('elu')(c1)

    c1 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(c1)

    c1 = Add()([c1, X_shortcut])

    p1 = MaxPooling2D((2, 2))(c1)

    ##########################################
    X_shortcut= Conv2D(16*n*2, (1, 1), padding='same', kernel_initializer='he_normal')(p1)

    #tower 1 (5X5)
    t11 = BatchNormalization()(X_shortcut)
    t11 = Activation('elu')(t11)
    t11 = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal')(t11)

    t11 = BatchNormalization()(t11)
    t11 = Activation('elu')(t11)
    t11 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(t11)

    t11 = BatchNormalization()(t11)
    t11 = Activation('elu')(t11)
    t11 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(t11)

    #tower 2 (3X3)
    t21 = BatchNormalization()(X_shortcut)
    t21 = Activation('elu')(t21)
    t21 = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal')(t21)

    t21 = BatchNormalization()(t21)
    t21 = Activation('elu')(t21)
    t21 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(t21)

    c2 = concatenate([t11, t21])

    c2 = Add()([c2, X_shortcut])

    p2 = MaxPooling2D((2, 2))(c2)

    ##########################################
    X_shortcut = Conv2D(32*n*2, (1, 1), padding='same', kernel_initializer='he_normal')(p2)

    #tower 1 (5X5)
    t12 = BatchNormalization()(X_shortcut)
    t12 = Activation('elu')(t12)
    t12 = Conv2D(32*n, (1, 1), padding='same', kernel_initializer='he_normal')(t12)

    t12 = BatchNormalization()(t12)
    t12 = Activation('elu')(t12)
    t12 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer='he_normal')(t12)

    t12 = BatchNormalization()(t12)
    t12 = Activation('elu')(t12)
    t12 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer='he_normal')(t12)

    #tower 2 (3X3)
    t22 = BatchNormalization()(X_shortcut)
    t22 = Activation('elu')(t22)
    t22 = Conv2D(32*n, (1, 1), padding='same', kernel_initializer='he_normal')(t22)

    t22 = BatchNormalization()(t22)
    t22 = Activation('elu')(t22)
    t22 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer='he_normal')(t22)

    c3 = concatenate([t12, t22])

    c3 = Add()([c3, X_shortcut])

    p3 = MaxPooling2D((2, 2))(c3)

    ###################Bridge#######################
    X_shortcut = Conv2D(128*n, (1, 1), padding='same',kernel_initializer='he_normal')(p3)

    t1b = BatchNormalization()(X_shortcut)
    t1b = Activation('elu')(t1b)
    t1b = Conv2D(64*n, (1, 1), padding='same', kernel_initializer='he_normal')(t1b)

    t1b = BatchNormalization()(t1b)
    t1b = Activation('elu')(t1b)
    t1b = Conv2D(64*n, (3, 3), padding='same', kernel_initializer='he_normal')(t1b)

    t1b = BatchNormalization()(t1b)
    t1b = Activation('elu')(t1b)
    t1b = Conv2D(64*n, (3, 3), padding='same', kernel_initializer='he_normal')(t1b)

    #tower 2 (3X3)
    t2b = BatchNormalization()(X_shortcut)
    t2b = Activation('elu')(t2b)
    t2b = Conv2D(64*n, (1, 1), padding='same', kernel_initializer='he_normal')(t2b)

    t2b = BatchNormalization()(t2b)
    t2b = Activation('elu')(t2b)
    t2b = Conv2D(64*n, (3, 3), padding='same', kernel_initializer='he_normal')(t2b)

    c4 = concatenate([t1b, t2b])

    c4 = Add()([c4, X_shortcut])

    ###################END BRIDGE#######################
    u5 = Conv2DTranspose(64*n, (2, 2), strides=(2, 2), padding='same') (c4)
    u5 = concatenate([u5, c3])

    u5 = Conv2D(64*n, (1, 1), padding='same', kernel_initializer='he_normal') (u5)

    X_shortcut = u5

    c5 = BatchNormalization()(X_shortcut)
    c5 = Activation('elu')(c5)
    c5 = Conv2D(64*n, (3, 3), padding='same', kernel_initializer='he_normal')(c5)

    c5 = BatchNormalization()(c5)
    c5 = Activation('elu')(c5)
    c5 = Conv2D(64*n, (3, 3), padding='same', kernel_initializer='he_normal')(c5)

    c5 = Add()([c5, X_shortcut])

    ################################################

    u6 = Conv2DTranspose(32*n, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c2])

    u6 = Conv2D(32*n, (1, 1), padding='same', kernel_initializer='he_normal') (u6)

    X_shortcut = u6

    c6 = BatchNormalization()(X_shortcut)
    c6 = Activation('elu')(c6)
    c6 = Conv2D(32*n, (3, 3), padding='same',kernel_initializer='he_normal')(c6)

    c6 = BatchNormalization()(c6)
    c6 = Activation('elu')(c6)
    c6 = Conv2D(32*n, (3, 3), padding='same',kernel_initializer='he_normal')(c6)

    c6 = Add()([c6, X_shortcut])



    u7 = Conv2DTranspose(16*n, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c1])
    u7 = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal') (u7)

    X_shortcut = u7

    c7 = BatchNormalization()(X_shortcut)
    c7 = Activation('elu')(c7)
    c7 = Conv2D(16*n, (3, 3), padding='same',kernel_initializer='he_normal')(c7)

    c7 = BatchNormalization()(c7)
    c7 = Activation('elu')(c7)
    c7 = Conv2D(16*n, (3, 3), padding='same',kernel_initializer='he_normal')(c7)

    c7 = Add()([c7, X_shortcut])

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c7)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    model.summary() 
    
    if map_weights == 'no':
    
        WeightedLoss = create_weighted_binary_crossentropy(1, 1.5)  
        
    elif map_weights == 'yes':

        WeightedLoss = create_weighted_binary_crossentropy_2(1, 1.5)     
    
    Adam = optimizers.Adam(lr=0.0008)  
    model.compile(optimizer=Adam, loss=WeightedLoss, metrics=[mean_iou, dice_coef])
    
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
    schedule = lrate.SGDRScheduler(min_lr=1e-6,
                             max_lr=3e-4,
                             steps_per_epoch=np.ceil(train_number/BATCH_SIZE),
                             lr_decay=0.9,
                             cycle_length=20,
                             mult_factor=1.5)

    ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3, verbose=1, 
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
                                  
                                  
def ResUnetIdentityDrop(train_generator, valid_generator, scheduler = 'Adam', n = 1, map_weights = 'no', model_name = 'ResUnetIdentity.h5'):
    
    n = n
    inputs = Input((None, None, 3))

    X_shortcut = Conv2D(8*n, (1, 1), padding='same',  kernel_initializer='he_normal')(inputs)
    
    c1 = Conv2D(8*n, (3, 3), padding='same',  kernel_initializer='he_normal')(X_shortcut)
    c1 = BatchNormalization()(c1)
    c1 = Activation('elu')(c1)

    c1 = Conv2D(8*n, (3, 3), padding='same', kernel_initializer='he_normal')(c1)

    c1 = Add()([c1, X_shortcut])

    p1 = MaxPooling2D((2, 2))(c1)

    ##########################################
    X_shortcut = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal')(p1)

    c2 = BatchNormalization()(X_shortcut)
    c2 = Activation('elu')(c2)
    c2 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c2)
    # c2 = Dropout(0.2)(c2)

    c2 = BatchNormalization()(c2)
    c2 = Activation('elu')(c2)
    c2 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c2)

    c2 = Add()([c2, X_shortcut])

    p2 = MaxPooling2D((2, 2))(c2)

    ##########################################
    X_shortcut = Conv2D(32*n, (1, 1), padding='same', kernel_initializer='he_normal')(p2)
    
    c3 = BatchNormalization()(X_shortcut)
    c3 = Activation('elu')(c3)
    c3 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer='he_normal')(c3)
    c3 = Dropout(0.1)(c3)

    c3 = BatchNormalization()(c3)
    c3 = Activation('elu')(c3)
    c3 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer='he_normal')(c3)

    c3 = Add()([c3, X_shortcut])

    p3 = MaxPooling2D((2, 2))(c3)

    ###################Bridge#######################
    X_shortcut = Conv2D(64*n, (1, 1), padding='same',kernel_initializer='he_normal')(p3)
    
    c4 = BatchNormalization()(X_shortcut)
    c4 = Activation('elu')(c4)
    c4 = Conv2D(64*n, (3, 3), padding='same',kernel_initializer='he_normal')(c4)
    c4= Dropout(0.1)(c4)

    c4 = BatchNormalization()(c4)
    c4 = Activation('elu')(c4)
    c4 = Conv2D(64*n, (3, 3), padding='same',kernel_initializer='he_normal')(c4)
    c4= Dropout(0.2)(c4)

    c4 = Add()([c4, X_shortcut])

    X_shortcut = c4

    c5 = BatchNormalization()(X_shortcut)
    c5 = Activation('elu')(c5)
    c5 = Conv2D(64*n, (3, 3), padding='same',kernel_initializer='he_normal')(c5)
    c5 = Dropout(0.1)(c5)

    c5 = BatchNormalization()(c5)
    c5 = Activation('elu')(c5)
    c5 = Conv2D(64*n, (3, 3), padding='same',kernel_initializer='he_normal')(c5) 
    c5 = Dropout(0.1)(c5)

    c5 = Add()([c5, X_shortcut])


    ###################END BRIDGE#######################


    u6 = Conv2DTranspose(32*n, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c3])

    u6 = Conv2D(32*n, (1, 1), padding='same', kernel_initializer='he_normal') (u6)

    X_shortcut = u6

    c6 = BatchNormalization()(X_shortcut)
    c6 = Activation('elu')(c6)
    c6 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer='he_normal')(c6)
    c6 = Dropout(0.1)(c6)

    c6 = BatchNormalization()(c6)
    c6 = Activation('elu')(c6)
    c6 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer='he_normal')(c6)

    c6 = Add()([c6, X_shortcut])

    ################################################

    u7 = Conv2DTranspose(16*n, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c2])
    u7 = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal') (u7)

    X_shortcut = u7

    c7 = BatchNormalization()(X_shortcut)
    c7 = Activation('elu')(c7)
    c7 = Conv2D(16*n, (3, 3), padding='same',kernel_initializer='he_normal')(c7)
    c7 = Dropout(0.1)(c7)

    c7 = BatchNormalization()(c7)
    c7 = Activation('elu')(c7)
    c7 = Conv2D(16*n, (3, 3), padding='same',kernel_initializer='he_normal')(c7)
    c7 = Dropout(0.1)(c7)

    c7 = Add()([c7, X_shortcut])

    u8 = Conv2DTranspose(8*n, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c1])
    u8 = Conv2D(8*n, (1, 1), padding='same', kernel_initializer='he_normal') (u8)

    X_shortcut = u8

    c8 = BatchNormalization()(X_shortcut)
    c8 = Activation('elu')(c8)
    c8 = Conv2D(8*n, (3, 3), padding='same',kernel_initializer='he_normal')(c8)     

    c8 = BatchNormalization()(c8)
    c8 = Activation('elu')(c8)
    c8 = Conv2D(8*n, (3, 3), padding='same',kernel_initializer='he_normal')(c8)
    c8 = Dropout(0.1)(c8) 

    c8 = Add()([c8, X_shortcut])

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c8)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    model.summary()

    if map_weights == 'no':
    
        WeightedLoss = create_weighted_binary_crossentropy(1, 1.5)  
        
    elif map_weights == 'yes':

        WeightedLoss = create_weighted_binary_crossentropy_2(1, 1.5)     
    
    Adam = optimizers.Adam(lr=0.0008)
    #model = multi_gpu_model(model, gpus=2)  
    model.compile(optimizer=Adam, loss=WeightedLoss, metrics=[mean_iou, dice_coef])
    
    #model = load_model(str(MODEL_CHECKPOINTS/'UnetSmall_v4.h5'), custom_objects={'mean_iou': mean_iou, 'dice_coef': dice_coef, 'weighted_binary_crossentropy': WeightedLoss})    
    
    checkpointer = ModelCheckpoint(str(MODEL_CHECKPOINTS/model_name), verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=30)
    tensorboard = tb.TrainValTensorBoard(log_dir=str(RESULTS_DIRECTORY / 'logs'), write_graph=True, write_images=True,
                                         batch_size = BATCH_SIZE, write_grads=False) 
    #tensorboard = TensorBoard(log_dir=str(RESULTS_DIRECTORY / 'logs'), histogram_freq=1, batch_size=BATCH_SIZE, write_graph=True, 
                              #write_grads=False, write_images=True, embeddings_freq=1, embeddings_layer_names=None, 
                              #embeddings_metadata=None, embeddings_data=None, update_freq='batch')
    plot_losses = plots.TrainingPlot()
    schedule = lrate.SGDRScheduler(min_lr=1e-6,
                             max_lr=3e-4,
                             steps_per_epoch=np.ceil(train_number/BATCH_SIZE),
                             lr_decay=0.9,
                             cycle_length=20,
                             mult_factor=1.5)

    ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1, 
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

def DeepResUnet(train_generator, valid_generator, scheduler = 'Adam', n = 1, map_weights = 'no', model_name = 'DeepResUnet.h5'):
    
    n = n

    inputs = Input((None, None, 3))
    X_shortcut = Conv2D(8*n, (1, 1), padding='same',  kernel_initializer='he_normal')(inputs)

    c1 = Conv2D(8*n, (5, 5), padding='same',  kernel_initializer='he_normal')(X_shortcut)
    c1 = BatchNormalization()(c1)
    c1 = Activation('elu')(c1)

    c1 = Conv2D(8*n, (3, 3), padding='same', kernel_initializer='he_normal')(c1)

    c1 = Add()([c1, X_shortcut])

    p1 = MaxPooling2D((2, 2))(c1)

    ##########################################
    X_shortcut = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal')(p1)

    c2 = BatchNormalization()(X_shortcut)
    c2 = Activation('elu')(c2)
    c2 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c2)

    c2 = BatchNormalization()(c2)
    c2 = Activation('elu')(c2)
    c2 = Conv2D(16*n, (3, 3), padding='same', kernel_initializer='he_normal')(c2)

    c2 = Add()([c2, X_shortcut])

    p2 = MaxPooling2D((2, 2))(c2)

    ##########################################
    X_shortcut = Conv2D(32*n, (1, 1), padding='same', kernel_initializer='he_normal')(p2)

    c3 = BatchNormalization()(X_shortcut)
    c3 = Activation('elu')(c3)
    c3 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer='he_normal')(c3)
    c3 = Dropout(0.2)(c3)

    c3 = BatchNormalization()(c3)
    c3 = Activation('elu')(c3)
    c3 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer='he_normal')(c3)

    c3 = Add()([c3, X_shortcut])

    p3 = MaxPooling2D((2, 2))(c3)

    ##############################################
    X_shortcut = Conv2D(64*n, (1, 1), padding='same', kernel_initializer='he_normal')(p3)

    c51 = BatchNormalization()(X_shortcut)
    c51 = Activation('elu')(c51)
    c51 = Conv2D(16*n, (3, 3),dilation_rate = (6,6),padding='same', kernel_initializer='he_normal')(c51)

    c52 = BatchNormalization()(X_shortcut)
    c52 = Activation('elu')(c52)
    c52 = Conv2D(16*n, (3, 3), dilation_rate = (12,12), padding='same', kernel_initializer='he_normal')(c52)

    c53 = BatchNormalization()(X_shortcut)
    c53 = Activation('elu')(c53)
    c53 = Conv2D(16*n, (3, 3), dilation_rate = (18,18), padding='same', kernel_initializer='he_normal')(c53)

    c54 = BatchNormalization()(X_shortcut)
    c54 = Activation('elu')(c54)
    c54 = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal')(c54)

    c5 = concatenate([c51,c52,c53,c54])
    c5 = SpatialDropout2D(0.2)(c5)

    c5 = Conv2D(64*n, (1, 1), padding='same', kernel_initializer='he_normal')(c5)

    c5 = Add()([c5, X_shortcut])
    #################################################################################

    u6 = Conv2DTranspose(32*n, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c3])
    u6 = Conv2D(32*n, (1, 1), padding='same', kernel_initializer='he_normal') (u6)

    X_shortcut = u6

    c6 = BatchNormalization()(X_shortcut)
    c6 = Activation('elu')(c6)
    c6 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer='he_normal')(c6)
    c6 = Dropout(0.1)(c6)

    c6 = BatchNormalization()(c6)
    c6 = Activation('elu')(c6)
    c6 = Conv2D(32*n, (3, 3), padding='same', kernel_initializer='he_normal')(c6)

    c6 = Add()([c6, X_shortcut])

    ################################################

    u7 = Conv2DTranspose(16*n, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c2])
    u7 = Conv2D(16*n, (1, 1), padding='same', kernel_initializer='he_normal') (u7)

    X_shortcut = u7

    c7 = BatchNormalization()(X_shortcut)
    c7 = Activation('elu')(c7)
    c7 = Conv2D(16*n, (3, 3), padding='same',kernel_initializer='he_normal')(c7)
    c7 = Dropout(0.1)(c7)

    c7 = BatchNormalization()(c7)
    c7 = Activation('elu')(c7)
    c7 = Conv2D(16*n, (3, 3), padding='same',kernel_initializer='he_normal')(c7)
    c7 = Dropout(0.1)(c7)

    c7 = Add()([c7, X_shortcut])

    #####################################################################################

    u8 = Conv2DTranspose(8*n, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c1])
    u8 = Conv2D(8*n, (1, 1), padding='same', kernel_initializer='he_normal') (u8)

    X_shortcut = u8

    c8 = BatchNormalization()(X_shortcut)
    c8 = Activation('elu')(c8)
    c8 = Conv2D(8*n, (3, 3), padding='same',kernel_initializer='he_normal')(c8)      

    c8 = BatchNormalization()(c8)
    c8 = Activation('elu')(c8)
    c8 = Conv2D(8*n, (3, 3), padding='same',kernel_initializer='he_normal')(c8)
    c8 = Dropout(0.1)(c8)

    c8 = Add()([c8, X_shortcut])

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c8)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    model.summary()

    if map_weights == 'no':
    
        WeightedLoss = create_weighted_binary_crossentropy(1, 1.5)  
        
    elif map_weights == 'yes':

        WeightedLoss = create_weighted_binary_crossentropy_2(1, 1.5)     
    
    Adam = optimizers.Adam(lr=0.001)  
    model.compile(optimizer=Adam, loss=WeightedLoss, metrics=[mean_iou, dice_coef])
    
    #model = load_model(str(MODEL_CHECKPOINTS/'UnetSmall_v4.h5'), custom_objects={'mean_iou': mean_iou, 'dice_coef': dice_coef, 'weighted_binary_crossentropy': WeightedLoss})    
    
    checkpointer = ModelCheckpoint(str(MODEL_CHECKPOINTS/model_name), verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=30)
    tensorboard = tb.TrainValTensorBoard(log_dir=str(RESULTS_DIRECTORY / 'logs'), write_graph=True, write_images=True,
                                         batch_size = BATCH_SIZE, write_grads=False) 
    #tensorboard = TensorBoard(log_dir=str(RESULTS_DIRECTORY / 'logs'), histogram_freq=1, batch_size=BATCH_SIZE, write_graph=True, 
                              #write_grads=False, write_images=True, embeddings_freq=1, embeddings_layer_names=None, 
                              #embeddings_metadata=None, embeddings_data=None, update_freq='batch')
    plot_losses = plots.TrainingPlot()
    schedule = lrate.SGDRScheduler(min_lr=1e-6,
                             max_lr=3e-4,
                             steps_per_epoch=np.ceil(train_number/BATCH_SIZE),
                             lr_decay=0.9,
                             cycle_length=20,
                             mult_factor=1.5)

    ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1, 
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
                                  

                                 




if __name__ == "__main__":


    train_generator, valid_generator = imageGenerator(color_mode='rgb')
   
    # InceptionUnetBasic(train_generator,valid_generator, scheduler = 'Adam', n = 2, map_weights = 'yes', model_name = 'ResUNeptionAtlasn2.h5')
    ResUnetIdentity(train_generator,valid_generator, scheduler = 'SGD',compiler = 'Adam',  n = 4, map_weights = 'yes', model_name = 'ResUnetCnafCore1N42w0_11SGD.h5')
    #ResUnetBig(train_generator,valid_generator, scheduler = 'Adam',n=1, map_weights = 'yes', model_name = 'ResUnetAtlasBigBridge.h5')
    #ResUnetSep(train_generator, valid_generator, scheduler = 'Adam', n = 1, map_weights = 'yes', model_name = 'ResUnetSepAtlas.h5')
