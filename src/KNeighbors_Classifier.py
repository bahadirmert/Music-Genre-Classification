#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Tue Feb  6 12:41:01 2018

    @ Author: Jose Jesus Torronteras Hernandez
    @ Github: https://github.com/xexuew
    @ Name: KNeighbors Classifier
    @ Description: 
"""


import numpy as np
import matplotlib.pyplot as plt
import sklearn
import time
import config
import itertools

from sklearn.neighbors import KNeighborsClassifier

#config.DATA_configuration.MULTIDIMENSIONAL_ARR = False # We want a 2D array Dimension

X_train = np.load(config.DATA_configuration.DATA['X_train'])
y_train = np.load(config.DATA_configuration.DATA['y_train'])
X_test = np.load(config.DATA_configuration.DATA['X_test'])
y_test = np.load(config.DATA_configuration.DATA['y_test'])

# Necessary to scale all the data

print("X_train dimensions: ", X_train.shape)
print("X_test dimensions: ", X_test.shape)

#X_train = np.reshape(X_train, (X_train.shape[0], -1))
#X_test = np.reshape(X_test, (X_test.shape[0], -1))
#print(X_train.shape, X_test.shape)

X_train =  sklearn.preprocessing.scale(X_train)
X_test =  sklearn.preprocessing.scale(X_test)


"""
    @ Description:
    @ Return: Nothing
"""
def confusion_matrix(cm, classes, title='Confusion matrix', cmap = plt.cm.Blues):
   
    plt.imshow(cm, interpolation='nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_cnf(model, dataset_x, dataset_y, GENRES):
   
    dataset_x = dataset_x
    dataset_y = dataset_y
    pred = model.predict(dataset_x)
    
    print("Real Test dataset labels: \n{}\n".format(dataset_y))
    print("Predicted Test dataset labels: \n{}".format(pred))

    cnf_matrix = sklearn.metrics.confusion_matrix(dataset_y, pred)
    plt.figure()
    confusion_matrix(cnf_matrix, classes = GENRES, title='Confusion matrix')
    plt.show()



GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Start Time
start_time = time.time()

print("Knn with n_neighbors = 1")

# Start with 1 Neighbors
knn = KNeighborsClassifier(n_neighbors = 1)
# Fit knn
knn.fit(X_train, y_train)

# Showing results
print("Training Score: {:.3f}".format(knn.score(X_train, y_train)))
print("Test score: {:.3f}".format(knn.score(X_test, y_test)))  
plot_cnf(knn, X_test, y_test, GENRES)
print("---- %s Seconds -----" % (time.time() - start_time))


# Array With all results
results_knn = []

# Start Time
start_time = time.time()

# Loop increments the number of neighbors
for i in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    results_knn.append(knn.score(X_test,y_test))
    print("With n_neighbors: ", i, " Score: ", knn.score(X_test,y_test))

# Get the max score obtained
max_accuracy_knn = max(results_knn)
best_k = 1+results_knn.index(max(results_knn))
print("Max Accuracy is {:.3f} on test dataset with {} neighbors.\n".format(max_accuracy_knn,best_k))

plt.plot(np.arange(1,11),results_knn)
plt.xlabel("n Neighbors")
plt.ylabel("Accuracy")
plt.show()

print("---- %s Seconds -----" % (time.time() - start_time))
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.fit_transform(X_test)