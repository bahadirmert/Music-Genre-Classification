#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Tue Feb  6 12:41:01 2018
    
    @ Author: Jose Jesus Torronteras Hernandez
    @ Github: https://github.com/xexuew
    @ Name: Extract Audio Features
    @ Description: 
"""

import librosa
import numpy as np
import csv
import os
import sys

class ExtractAudioFeatures(object):

    def __init__(self, config):
        
        self.PATH = config['PATH_CONFIGURATION']['AUDIO_PATH']
        self.DEST = config['PATH_CONFIGURATION']['NUMPY_PATH']

        self.SR = int(config['AUDIO_FEATURES']['SR'])
        self.N_MELS = int(config['AUDIO_FEATURES']['N_MELS'])
        self.N_FFT = int(config['AUDIO_FEATURES']['N_FFT'])
        self.HOP_LENGTH = int(config['AUDIO_FEATURES']['HOP_LENGTH'])
        self.DURATION = int(config['AUDIO_FEATURES']['DURATION'])
    
    #
    # Description:
    # Input:
    # Output:
    def librosaAudio(self, file_Path):
        
        # Load audio file with Librosa
        y, sr = librosa.load(file_Path, duration = self.DURATION)

        S = librosa.power_to_db(
                librosa.feature.melspectrogram(
                    y,
                    sr = sr,
                    n_mels = self.N_MELS,
                    n_fft = self.N_FFT,
                    hop_length = self.HOP_LENGTH),
                ref_power = np.max)
        
        return S

    #
    # Description:
    # Input:
    # Output:
    def prepossessingAudio(self):
        
        directory = ['blues', 'blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        progress = 1.0

        os.system('mkdir data/')
        os.system('mkdir ' + self.DEST) # mkdir data/songs_np/ # data/songs_np/blues data/songs_np/classical
        
        with open(self.DEST + "url.csv", 'w') as csvfile:
            writer = csv.writer(csvfile)

            for root, subdirs, files in os.walk(self.PATH):
                subdirs.sort() # Sort all subdirs
                
                os.system('mkdir ' + self.DEST + directory[0]) # Create the folder to save the melspectrogram
                
                for filename in files:
                    
                    if filename.endswith('.au'): # Only we want audio files, delete files like .DS_store
                        
                        file_Path = os.path.join(root, filename) # Get the Audio file path
                        print('File %s (full path: %s)' % (filename, file_Path))
                        
                        audio_ID = os.path.splitext(filename)[0].replace(".", "_")  # Get ID audio
                        
                        try:

                            S = self.librosaAudio(file_Path) # Get the Melspectrogram
                            path_song = self.DEST + directory[0] + "/" + audio_ID

                            np.save(path_song, S)

                            writer.writerow([directory[0], path_song + ".npy"])
                        except Exception as e:
                            
                            print("Error accured" + str(e))

                        porcentaje = progress / 10
                        sys.stdout.write("\n%f%%  " % porcentaje)
                        sys.stdout.flush()
                        progress += 1
                            
                directory.pop(0) # Next directory
            del directory, porcentaje, progress, file_Path, path_song
            del files, root, subdirs
            del S