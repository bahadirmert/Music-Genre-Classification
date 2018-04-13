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
import os
import config
from sklearn.model_selection import train_test_split



"""
    @ Description: This function Download from MongoDB Database all songs Melspectrogram arrays. Loop range is 100, because we have 100 songs.
    @ Return: The Features of each gender.
"""
def get_features_online( nameDB):
    
    songs_db = config.Connection.return_data(nameDB)
    aux_list = []
        
    try:
        for i in range(100):
            
            arr_aux = np.array(songs_db[i].get(list(songs_db[i].keys())[1])) # 1 is the first element bd => songs_doc[1]
            aux_list.append(arr_aux)       
        
        # if MULTIDIMENSIONAL_ARR is True we want a 3D array (*, * , *) for Neural Network
        if config.DATA_configuration.MULTIDIMENSIONAL_ARR:
            features_arr = aux_list
        else:
            features_arr = np.vstack(aux_list)

    except Exception as e:
        print ("Error occurred" + str(e))
          
    return features_arr



"""
    @ Description: This function Get Numpy files neccessary to generate all the train and test files.
    @ Return: The Features of each gender.
"""
def get_features_local( name):
    
    aux_list = []
    limit = 0
    print("Getting.." + config.PATH_MUSIC_NP_FILES + name)
    
    for root, subdirs, files in os.walk(config.PATH_MUSIC_NP_FILES + name):
        
        for filename in files:
            file_Path = os.path.join(root, filename) # Get the Audio file path

            try:
                arr_aux = np.load(file_Path)
                
            except UnicodeDecodeError as e:
                print(filename)
                print ("Error occurred" + str(e))
            
            limit = limit +1
            if limit == config.DATA_configuration.DATA_SIZE: # See config.py -> DATA_SIZE
                break
            
            aux_list.append(arr_aux)
        
        # if MULTIDIMENSIONAL_ARR is True we want a 3D array (*, * , *) for Neural Network
        if config.DATA_configuration.MULTIDIMENSIONAL_ARR:
            features_arr = aux_list
        else:
            features_arr = np.vstack(aux_list)
            
    return features_arr 
    

# -------- Main Program ----------- #
    
if config.Connection.ALLOWED_CONNECTION:
    arr_blues = get_features_online('blues')
    arr_classical = get_features_online('classical')
    arr_country = get_features_online('country')
    arr_disco = get_features_online('disco')
    arr_hiphop = get_features_online('hiphop')
    arr_jazz = get_features_online('jazz')
    arr_metal = get_features_online('metal')
    arr_pop = get_features_online('pop')
    arr_reggae = get_features_online('reggae')
    arr_rock = get_features_online('rock')
else:
    arr_blues = get_features_local('blues')
    arr_classical = get_features_local('classical')
    arr_country = get_features_local('country')
    arr_disco = get_features_local('disco')
    arr_hiphop = get_features_local('hiphop')
    arr_jazz = get_features_local('jazz')
    arr_metal = get_features_local('metal')
    arr_pop = get_features_local('pop')
    arr_reggae = get_features_local('reggae')
    arr_rock = get_features_local('rock')


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


"""
 With train_test_split() it is more easier obtain the necessary elements for the later learning.
 We can change the size in the config file.
"""
print("test-size = " + str(config.DATA_configuration.TRAIN_TEST_SPLIT_SIZE) + " Change value in config.py")
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = config.DATA_configuration.TRAIN_TEST_SPLIT_SIZE , random_state = 0)

print("X_train Tamaño: %s - X_test Tamaño: %s - y_train Tamaño: %s - y_test Tamaño: %s" % (X_train.shape, X_test.shape, y_train.shape, y_test.shape))

os.system('mkdir ' + str(config.DATA_configuration.PATH_DATA_FILES))
np.save(config.DATA_configuration.DATA['X_train'], X_train)
np.save(config.DATA_configuration.DATA['X_test'], X_test)
np.save(config.DATA_configuration.DATA['y_train'], y_train)
np.save(config.DATA_configuration.DATA['y_test'], y_test)