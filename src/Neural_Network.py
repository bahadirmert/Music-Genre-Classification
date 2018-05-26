#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 12:41:01 2018

@author: josetorronteras
@name:
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras import backend as K


class CNNModel(object):

    def __init__(self, config, X):
        self.filters = 32 # number of convolutional filters to use
        self.pool_size = (2, 2)  # size of pooling area for max pooling
        self.kernel_size = (3, 3)  # convolution kernel size
        self.nb_layers = 4
        self.input_shape = (128, 625, 1) # cambiar por x.shape
        
        
    def build_model(self, nb_classes):
    
        model = Sequential()
        model.add(
                Conv2D(
                    self.filters,
                    self.kernel_size,
                    padding ='same',
                    input_shape = self.input_shape))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        
        model.add(
                Conv2D(
                    self.filters,
                    self.kernel_size))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = self.pool_size))
        model.add(Dropout(0.25))
        
        model.add(
            Conv2D(
                self.filters + 32,
                self.kernel_size,
                padding ='same'))
        model.add(Activation('relu'))
        
        model.add(
            Conv2D(
                self.filters + 32,
                self.kernel_size,
                padding ='same'))
        model.add(MaxPooling2D(pool_size = self.pool_size))
        model.add(Dropout(0.25))
             
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation("softmax")) #mirar
        
        return model
