#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 12:41:01 2018

@author: Jose Jesus Torronteras Hernandez
@github: https://github.com/xexuew
@name: TSNE figure
@description: 
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import config

print("TSNE")
print("Esto tardara bastante, Desea cargar un fichero ya existente?, Y/N")

while True:
    option = input()
    
    if option == 'Y' or option == 'y':
        X_embedded = np.load(config.CSV_configuration.CSV_DICT['arr_tsne'])
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], cmap=plt.cm.Spectral)
        plt.savefig(config.SAVE_PLT+'/tsne_fig.png')
        plt.show()     
        break
    
    elif option == 'N' or option == 'n':
        
        try:
            X_train = np.loadtxt(config.CSV_configuration.CSV_DICT['X_train'], delimiter=',')
            print("Starting TSNE..")
            X_embedded = TSNE(n_components=2).fit_transform(X_train)
            np.save(config.CSV_configuration.CSV_DICT['arr_tsne'])
            plt.scatter(X_embedded[:, 0], X_embedded[:, 1], cmap=plt.cm.Spectral)
            plt.savefig(config.SAVE_PLT+'/tsne_fig.png')
            plt.show()
        except FileNotFoundError:
            print("File: " + config.CSV_configuration.CSV_DICT['X_train'] + " Not found")
            print("Please Execute Get_Train_Test_Data.py before")
            
        break
    
    else:
        print("Please enter Y/N")
