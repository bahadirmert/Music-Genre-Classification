#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 12:41:01 2018

@author: Jose Jesus Torronteras Hernandez
@github: https://github.com/xexuew
@name: Extract Audio Features
@description: 
"""

import sys, os, librosa, urllib.request
import config


directory = ['blues', 'blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
progress = 1.0

"""
 Check If GTZAN database is downloaded, otherwise we will download and then unzip in the indicated folder.
"""    
if not os.path.isdir(config.PATH_MUSIC):
    print("Music not found in: " + config.PATH_MUSIC + " - Please change config.py")

    print("Downloading with urllib: " + config.PATH_MUSIC_URL)
    urllib.request.urlretrieve(config.PATH_MUSIC_URL, "GTZAN.tar.gz")
    
    print("Uncompress")
    if(os.path.isdir('/app/data')):
        os.system('mkdir /app/data')
    os.system('tar -zxvf GTZAN.tar.gz -C /app/data/')


def prepossessingAudio(file_Path, audio_Id):
    data_to_insert = {}
        
    y, sr = librosa.load(file_Path, duration= 30.00 , mono=True) # Load audio file with Librosa
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels= 128, n_fft= 2048, hop_length=1024)
    
    data_to_insert[audio_Id] = S.tolist() 
    songs.insert_one(data_to_insert) # Insert in MongoDB


"""
 We go throught folders and establish a connection with MongoDB.
 We call prepossessingAudio() function to get the melspectrogram with Librosa module.
"""
for root, subdirs, files in os.walk(config.PATH_MUSIC):

    subdirs.sort() # Sort all subdirs
    
    client, songs = config.Connection.connection_DB(directory[0]) # Connection with MongoDB
    
    songs.delete_many({}) # Delete all elements in the Database
    
    for filename in files:
        
        if filename.endswith('.au'): # Only we want audio files, delete files like .DS_store
            
            file_Path = os.path.join(root, filename) # Get the Audio file path
            print('Fichero %s (full path: %s)' % (filename, file_Path))
            
            audio_ID = os.path.splitext(filename)[0].replace(".", "_")  # Get ID audio necessary for MongoDB
            try:
                prepossessingAudio(file_Path, audio_ID)
                
            except Exception as e:
                print("Error accured" + str(e))

        if filename.endswith('au'):
            porcentaje = progress/10
            sys.stdout.write("\n%f%%  " % porcentaje)
            sys.stdout.flush()
            progress += 1
        
        if progress == 10 or progress == 20 or progress == 30 or progress == 40 or progress == 50 or progress == 60 or \
        progress == 70 or progress == 80 or progress == 90 or progress == 100:
            break
            
    client.close() # Close connection with the database
    directory.pop(0) # Next directory

