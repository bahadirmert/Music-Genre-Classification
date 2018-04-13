#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 12:41:01 2018

@author: josetorronteras
@name: Config
"""
from pymongo import MongoClient

class Audio_features:
    SR = 22050 # default see https://librosa.github.io/librosa/generated/librosa.core.load.html
    # Generate (128, 625)
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 1024
    DURATION = 29.02

class Connection:
    
    ALLOWED_CONNECTION = False
    DATA_BASE_DICT = {
        'blues': 'ds219318.mlab.com:19318/blues_db',
        'classical': 'ds119258.mlab.com:19258/classical_db',
        'country': 'ds119268.mlab.com:19268/country_db',
        'disco': 'ds119268.mlab.com:19268/disco_db',
        'hiphop': 'ds159187.mlab.com:59187/hiphop_db',
        'jazz': 'ds119258.mlab.com:19258/jazz_db',
        'metal': 'ds119988.mlab.com:19988/metal_db',
        'pop': 'ds119258.mlab.com:19258/pop_db',
        'reggae': 'ds121088.mlab.com:21088/reggae_db',
        'rock': 'ds119268.mlab.com:19268/rock_db'
    }
    
    DATA_BASE_NAME_DICT = {
        'blues': 'blues_db',
        'classical': 'classical_db',
        'country': 'country_db',
        'disco': 'disco_db',
        'hiphop': 'hiphop_db',
        'jazz': 'jazz_db',
        'metal': 'metal_db',
        'pop': 'pop_db',
        'reggae': 'reggae_db',
        'rock': 'rock_db'
    }
    
    def connection_DB( nameDB):
        
        data_base_url = "mongodb://user:test@" + Connection.DATA_BASE_DICT[nameDB]
        print("Connect to: ", data_base_url)

        client = MongoClient( data_base_url)
        db = client[Connection.DATA_BASE_NAME_DICT[nameDB]]
        songs = db['db']
        
        return client, songs
    
    def return_data( nameDB, tam = 100):
        client, songs = Connection.connection_DB(nameDB)
        songs_doc = list(songs.find()[:tam])
        
        return songs_doc


class DATA_configuration:

    PATH_DATA_FILES = './data/data_files/'
    DATA = {
        'X_train' : './data/data_files/X_train.npy',
        'X_test' : './data/data_files/X_test.npy',
        'y_train' : './data/data_files/y_train.npy',
        'y_test' : './data/data_files/y_test.npy',
        'arr_tsne' : './data/arrays/arr_TSNE.npy'
    }
    MULTIDIMENSIONAL_ARR = False # Change to False if you dont want a 3D Array.
    DATA_SIZE = 100 # 20% of dataset
    TRAIN_TEST_SPLIT_SIZE = 0.2 # Size of the X_test
    
    
PATH_MUSIC = './data/genres/'
PATH_MUSIC_NP_FILES = './data/songs_np/'
PATH_MUSIC_URL = 'http://opihi.cs.uvic.ca/sound/genres.tar.gz'
SAVE_PLT = './data/assets'
