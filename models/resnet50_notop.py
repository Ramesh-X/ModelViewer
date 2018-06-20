'''
Created on Feb 1, 2018

@author: rameshpr
'''
import os
import numpy as np
import cv2 as cv
import time

from keras import layers
from keras.layers import Input,Dense,Activation,Flatten,Conv2D,MaxPooling2D,AveragePooling2D,GlobalAveragePooling2D,GlobalMaxPooling2D,BatchNormalization,Dropout
from keras.models import Model
from keras import backend as K
from numpy.linalg import norm

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from libs.common import FEATURE_TYPE

class resnet50_notop(object):
    '''
    classdocs
    '''

    def __init__(self,input_shape=(224,112,3), weight_path=None, feature_type=FEATURE_TYPE.GLOBAL_FEATURES):
        '''
        Constructor
        '''
        self.model = None
        self.input_shape = input_shape
        self.weight_path = weight_path
        self.predict_time_ms = 0
        self.feature_type = feature_type
        self._compile()
            
        
    def _identity_block(self,input_tensor, kernel_size, filters, stage, block):
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
    
        x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)
    
        x = Conv2D(filters2, kernel_size,
                   padding='same', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)
    
        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    
        x = layers.add([x, input_tensor])
        x = Activation('relu')(x)
        return x


    def _conv_block(self,input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
    
        x = Conv2D(filters1, (1, 1), strides=strides,
                   name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)
    
        x = Conv2D(filters2, kernel_size, padding='same',
                   name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)
    
        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    
        shortcut = Conv2D(filters3, (1, 1), strides=strides,
                          name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    
        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        return x

    def _compile(self):
                
        bn_axis=3
        inputs = Input(shape=self.input_shape)
        x = Conv2D(
            64, (7, 7), strides=(2, 2), padding='same', name='conv1')(inputs)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 1))(x)
    
        x = self._conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self._identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self._identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    
        x = self._conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = self._identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self._identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = self._identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    
        x = self._conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = self._identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = self._identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = self._identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = self._identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = self._identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    
        x = self._conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = self._identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = self._identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        if self.feature_type == FEATURE_TYPE.GLOBAL_FEATURES:
            x = AveragePooling2D((7,7), name='avg_pool') (x)
            x = Flatten() (x)
        
        model = Model(inputs, x)
        if self.weight_path is not None:
            model.load_weights(self.weight_path)
        print model.summary()
        self.model = model
        
    def predict(self,x,featurewise_normalize = False, samplewise_normalize=False):
        img_rz = self.preprocess(x, featurewise_normalize, samplewise_normalize)
        img_rz = np.expand_dims(img_rz, axis=0)
        pred = self.model.predict(img_rz)
        if self.feature_type == FEATURE_TYPE.GLOBAL_FEATURES:
            pred = pred/norm(pred)
            return pred
        else:
            pred = pred.mean(axis=2)
            return pred[0]

    def preprocess(self, img, featurewise_normalize = False, samplewise_normalize=False):
        img = cv.resize(img,(self.input_shape[1],self.input_shape[0])).astype(np.float32)
        if featurewise_normalize:
            img = self.featurewise_norm(img)
        elif samplewise_normalize:
            img = self.samplewise_norm(img)
        else:
            img[:, :, 0] -= 103.939
            img[:, :, 1] -= 116.779
            img[:, :, 2] -= 123.68
        
        return img

    def samplewise_norm(self, img):
        print "Samplewise mean"
        mean = img.mean(axis = (0, 1))
        std = img.std(axis=(0, 1))
        img = (img - mean) / std
        return img 
    
    def featurewise_norm(self, img):
        print "Featurewise norm"
        mean = np.array([98.08817463, 93.85825656, 97.5424934])
        std = np.array([56.67358797, 55.0811798, 55.38122459])

        img = (img - mean) / std

        return img
