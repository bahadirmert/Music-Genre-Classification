#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Tue Feb  6 12:41:01 2018
    
    @ Author: Jose Jesus Torronteras Hernandez
    @ Github: https://github.com/xexuew
    @ Name: Extract Audio Features
    @ Description: 
"""

import sys
import os
import librosa
import urllib.request 
import config
import numpy as np



"""
    @ Description: This Function check if GTZAN dataset is dowloaded and uncompressed. Otherwise we will download and then unzip in the indicated folder.
    @ Return: Nothing
"""
def check_files():
    
    if not os.path.isdir(config.PATH_MUSIC):
        
        os.system('mkdir ' + config.PATH_MUSIC)
        print("Music not found in: " + config.PATH_MUSIC + " - Please change config.py")
    
        print("Downloading with urllib: " + config.PATH_MUSIC_URL)
        urllib.request.urlretrieve(config.PATH_MUSIC_URL, "GTZAN.tar.gz")
        
        print("Uncompress GTZAN.tar.gz")
        if(os.path.isdir('./data/')):
            os.system('mkdir ./data/')
        os.system('tar -zxvf GTZAN.tar.gz -C ./data/')
        
    else:
        
        print("GTZAN dataset founded " + config.PATH_MUSIC)


"""
    @ Description: This Function extract all the songs features. Its necessary Librosa.
    @ Returns: (Array) Melspectrogram Song.
"""
def prepossessingAudio(file_Path, audio_Id):
    
    y, sr = librosa.load(file_Path, duration=config.Audio_features.DURATION) # Load audio file with Librosa
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=config.Audio_features.N_MELS, n_fft= config.Audio_features.N_FFT, hop_length = config.Audio_features.HOP_LENGTH) #S = librosa.feature.melspectrogram(y, sr=sr, n_mels= 128, n_fft= 2048, hop_length=1024)   
    
    return S
    

"""
    Main Program

"""
directory = ['blues', 'blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
progress = 1.0

check_files()

for root, subdirs, files in os.walk(config.PATH_MUSIC):
    subdirs.sort() # Sort all subdirs
    
    if config.Connection.ALLOWED_CONNECTION:
        
        data_to_insert = {}
        
        client, songs = config.Connection.connection_DB(directory[0]) # Connection with MongoDB
    
        songs.delete_many({}) # Delete all elements in the Database
        
        for filename in files:
        
            if filename.endswith('.au'): # Only we want audio files, delete files like .DS_store
                
                file_Path = os.path.join(root, filename) # Get the Audio file path
                print('File %s (full path: %s)' % (filename, file_Path))
                
                audio_ID = os.path.splitext(filename)[0].replace(".", "_")  # Get ID audio necessary for MongoDB
                
                try:
                    S = prepossessingAudio(file_Path, audio_ID)
                    data_to_insert[audio_ID] = S.tolist() 
                    songs.insert_one(data_to_insert) # Insert in MongoDB
                except Exception as e:
                    print("Error accured" + str(e))

                porcentaje = progress / 10
                sys.stdout.write("\n%f%%  " % porcentaje)
                sys.stdout.flush()
                progress += 1
        
        client.close() # Close connection with the database
        directory.pop(0) # Next directory
        
    else:
        
        os.system('mkdir ./data/songs_np/')
        path_dir = "./data/songs_np/" + directory[0]
        os.system('mkdir ' + str(path_dir)) # Create the folder to save the melspectrogram
        
        for filename in files:
            
            if filename.endswith('.au'): # Only we want audio files, delete files like .DS_store
                
                file_Path = os.path.join(root, filename) # Get the Audio file path
                print('File %s (full path: %s)' % (filename, file_Path))
                
                audio_ID = os.path.splitext(filename)[0].replace(".", "_")  # Get ID audio
                
                try:
                    
                    S = prepossessingAudio(file_Path, audio_ID)
                    path_song = "./data/songs_np/" + directory[0] + "/" + audio_ID

                    np.save(path_song, S)
                    
                except Exception as e:
                    
                    print("Error accured" + str(e))

                porcentaje = progress / 10
                sys.stdout.write("\n%f%%  " % porcentaje)
                sys.stdout.flush()
                progress += 1
                
        directory.pop(0) # Next directory