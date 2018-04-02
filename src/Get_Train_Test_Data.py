#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 12:41:01 2018

@author: Jose Jesus Torronteras Hernandez
@github: https://github.com/xexuew
@name: Get Train Test Data
@description: 
"""

import numpy as np
from sklearn.model_selection import train_test_split
import config

"""
 Download from Database all songs arrays.
 Loop range is 100, because we have 100 songs.
"""
def get_train_test( nameDB):

    mfcc_aux = config.Connection.return_data(nameDB)
    aux_list = []
        
    try:
        for i in range(100):
            arr_aux = np.array(mfcc_aux[i].get(list(mfcc_aux[i].keys())[1])) # 1 es el primer elemento de la bd => songs_doc[1]
            aux_list.append(arr_aux)       
        mfcc_aux = np.vstack(aux_list)

    except Exception as e:
        print ("Error occurred" + str(e))
          
    return mfcc_aux


arr_blues = get_train_test('blues')
arr_classical = get_train_test('classical')
arr_country = get_train_test('country')
arr_disco = get_train_test('disco')
arr_hiphop = get_train_test('hiphop')
arr_jazz = get_train_test('jazz')
arr_metal = get_train_test('metal')
arr_pop = get_train_test('pop')
arr_reggae = get_train_test('reggae')
arr_rock = get_train_test('rock')

# All songs arrays
features = np.vstack((arr_blues, arr_classical, arr_country, arr_disco, arr_hiphop, arr_jazz, arr_metal, arr_pop, arr_reggae, arr_rock))

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

"""
 With train_test_split() it is more easier obtain the necessary elements for the later learning.
 We can change the size in the config file.
"""
print("test-size = " + str(config.CSV_configuration.TRAIN_TEST_SPLIT_SIZE) + " Change value in config.py")
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = config.CSV_configuration.TRAIN_TEST_SPLIT_SIZE , random_state = 0)

print("X_train Tamaño: %s - X_test Tamaño: %s - y_train Tamaño: %s - y_test Tamaño: %s" % (X_train.shape, X_test.shape, y_train.shape, y_test.shape))

np.savetxt(config.CSV_configuration.CSV_DICT['X_train'], X_train, delimiter=',')
np.savetxt(config.CSV_configuration.CSV_DICT['X_test'], X_test, delimiter=',')
np.savetxt(config.CSV_configuration.CSV_DICT['y_train'], y_train, delimiter=',')
np.savetxt(config.CSV_configuration.CSV_DICT['y_test'], y_test, delimiter=',')
