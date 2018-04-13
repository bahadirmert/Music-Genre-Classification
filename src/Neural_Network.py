#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 12:41:01 2018

@author: josetorronteras
@name:
"""
import numpy as np
import config
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.utils import np_utils
from keras import backend as K
from os.path import isfile
K.set_image_dim_ordering('th')



X_train = np.load(config.DATA_configuration.DATA_DICT['X_train'])
y_train = np.load(config.DATA_configuration.DATA_DICT['y_train'])
X_test = np.load(config.DATA_configuration.DATA_DICT['X_test'])
y_test = np.load(config.DATA_configuration.DATA_DICT['y_test'])

"""
y_train = pd.get_dummies(y_train).values
y_test = pd.get_dummies(y_test).values
"""

print("Working..")


#X_train =  sklearn.preprocessing.scale(X_train)
#X_test =  sklearn.preprocessing.scale(X_test)

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

"""
fig = plt.figure(figsize=(10,7))
for i in range(25):
    fig.add_subplot(5, 5 ,(i+1), xticks=[], yticks=[])
    plt.title("{}".format(GENRES[int(y_train[i])]))
    plt.imshow(X_train[i])
plt.savefig('X_train.jpg')
"""


X_train = X_train.reshape(X_train.shape[0], 1, 128, 625).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 128, 625).astype('float32')

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def build_model(X, Y ,nb_classes):
    filters = 32  # number of convolutional filters to use
    pool_size = (2, 2)  # size of pooling area for max pooling
    kernel_size = (3, 3)  # convolution kernel size
    nb_layers = 4
    input_shape = (1, X.shape[2], X.shape[3])

    model = Sequential()
    model.add(Conv2D(filters, kernel_size, padding ='valid', input_shape = input_shape))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    for layer in range(nb_layers-1):
        model.add(Conv2D(filters, kernel_size))
        model.add(BatchNormalization(axis=1))
        model.add(ELU(alpha=1.0))  
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    return model

def baseline_model():
	# create model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),activation='relu',input_shape=(1, 128, 625)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.01), metrics=['accuracy'])
    return model

model = build_model(X_train,y_train, nb_classes=len(GENRES))
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
model.summary()

load_checkpoint = True
checkpoint_filepath = 'weights.hdf5'
if (load_checkpoint):
    print("Looking for previous weights...")
    if ( isfile(checkpoint_filepath) ):
        print ('Checkpoint file detected. Loading weights.')
        model.load_weights(checkpoint_filepath)
    else:
        print ('No checkpoint file detected.  Starting from scratch.')
else:
    print('Starting from scratch (no checkpoint)')
checkpointer = ModelCheckpoint(filepath=checkpoint_filepath, verbose=1, save_best_only=True)

batch_size = 128
nb_epoch = 100
model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
  verbose=1, validation_data=(X_test, y_test), callbacks=[checkpointer])
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])