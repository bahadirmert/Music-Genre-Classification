#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 12:41:01 2018

@author: josetorronteras
@name:
"""

import numpy as np, matplotlib.pyplot as plt
import pandas as pd
import config
import sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

X_train = np.load('./data/arrays/X_train.npy')
y_train = np.loadtxt(config.CSV_configuration.CSV_DICT['y_train'], delimiter=',')
X_test = np.loadtxt(config.CSV_configuration.CSV_DICT['X_test'], delimiter=',')
y_test = np.loadtxt(config.CSV_configuration.CSV_DICT['y_test'], delimiter=',')


y_train = pd.get_dummies(y_train).values
y_test = pd.get_dummies(y_test).values

print("Working..")

model = Sequential()

model.add(Dense(128, input_dim=646, activation='tanh'))
model.add(Dense(40, activation='tanh'))
model.add(Dense(10, activation='softmax'))

model.compile(Adam(lr=0.01),'categorical_crossentropy',metrics=['accuracy'])

model.summary()

results = model.fit(X_train,y_train,epochs=200, batch_size=128)


y_pred = model.predict(X_test)

y_test_class = np.argmax(y_test,axis=1)
y_pred_class = np.argmax(y_pred,axis=1)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test_class,y_pred_class))
print(confusion_matrix(y_test_class,y_pred_class))

plt.plot(results.history['loss'])
plt.show()
