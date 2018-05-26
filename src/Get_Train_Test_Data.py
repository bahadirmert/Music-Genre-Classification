#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Tue Feb  6 12:41:01 2018
    
    @ Author: Jose Jesus Torronteras Hernandez
    @ Github: https://github.com/xexuew
    @ Name: Get Train Test Data
    @ Description: 
"""

import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class GetTrainTestData(object):
    
    def __init__(self, config):
        
        self.PATH = config['PATH_CONFIGURATION']['NUMPY_PATH']

        self.SIZE = int(config['DATA_CONFIGURATION']['DATA_SIZE'])
        self.SPLIT_SIZE = float(config['DATA_CONFIGURATION']['SPLIT_SIZE'])
        self.MULTIDIM = int(config['DATA_CONFIGURATION']['MULTIDIMENSIONAL_ARR'])

    #
    # Description:
    # Input:
    # Output:
    def get_features(self, genre):

        aux_list = []
        limit = 0
        print("Getting.." + self.PATH + genre)
        
        with open(self.PATH + "url.csv") as f:

            for key, path in csv.reader(f):

                if key == genre:
                    try:
                        arr_aux = np.load(path)
                        
                    except UnicodeDecodeError as e:
                        print(path)
                        print ("Error occurred" + str(e))
                    
                    limit = limit + 1
                    if limit == self.SIZE: # See config.py -> DATA_SIZE
                        break
                    
                    aux_list.append(arr_aux)
                
            # if MULTIDIMENSIONAL_ARR is True we want a 3D array (*, * , *) for Neural Network
            if self.MULTIDIM:
                features_arr = aux_list
            else:
                features_arr = np.vstack(aux_list)
        
        return features_arr 

    #
    # Description:
    # Input:
    # Output:
    def split_dataset(self):
        
        arr_blues = self.get_features('blues')
        arr_classical = self.get_features('classical')
        arr_country = self.get_features('country')
        arr_disco = self.get_features('disco')
        arr_hiphop = self.get_features('hiphop')
        arr_jazz = self.get_features('jazz')
        arr_metal = self.get_features('metal')
        arr_pop = self.get_features('pop')
        arr_reggae = self.get_features('reggae')
        arr_rock = self.get_features('rock')

        # All songs arrays
        features = np.vstack((arr_blues,\
                            arr_classical,\
                            arr_country,\
                            arr_disco,\
                            arr_hiphop,\
                            arr_jazz,\
                            arr_metal,\
                            arr_pop,\
                            arr_reggae,\
                            arr_rock))


        # Labels that identifies the musical genre
        labels = np.concatenate((np.zeros(len(arr_blues)),\
                                np.ones(len(arr_classical)),\
                                np.full(len(arr_country), 2),\
                                np.full(len(arr_disco), 3),\
                                np.full(len(arr_hiphop), 4),\
                                np.full(len(arr_jazz), 5),\
                                np.full(len(arr_metal), 6),\
                                np.full(len(arr_pop), 7),\
                                np.full(len(arr_reggae), 8),\
                                np.full(len(arr_rock), 9)))

        # Transforms features by scaling each feature to a given range.
        features = MinMaxScaler().fit_transform(features.reshape(-1, 625)).reshape(features.shape[0], 128, 625)

        # With train_test_split() it is more easier obtain the necessary elements for the later learning.
        print("test-size = " + str(self.SPLIT_SIZE) + " Change value in config.py") # We can change the size in the config file.
        print("data-size = " + str(self.SIZE) + " Change value in config.py") # We can change the size in the config file.

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size = self.SPLIT_SIZE,
            random_state = 0,
            stratify = labels)

        print("X_train Tamaño: %s - X_test Tamaño: %s - y_train Tamaño: %s - y_test Tamaño: %s" % (X_train.shape, X_test.shape, y_train.shape, y_test.shape))

        return X_train, X_test, y_train, y_test
    