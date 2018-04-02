#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 12:41:01 2018

@author: josetorronteras
@name: Config
"""
from pymongo import MongoClient

class Connection:
    
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


class CSV_configuration:

    CSV_DICT = {
        'X_train' : '../data/csv_files/X_train.csv',
        'X_test' : '../data/csv_files/X_test.csv',
        'y_train' : '../data/csv_files/y_train.csv',
        'y_test' : '../data/csv_files/y_test.csv',
        'arr_tsne' : '../data/arrays/arr_TSNE.npy'
    }
    
    TRAIN_TEST_SPLIT_SIZE = 0.2
    
    
PATH_MUSIC = '../data/genres'
PATH_MUSIC_URL = 'http://opihi.cs.uvic.ca/sound/genres.tar.gz'
SAVE_PLT = '../data/assets'
