#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 12:41:01 2018

@author: Jose Jesus Torronteras Hernandez
@github: https://github.com/xexuew
@name: KNeighbors Classifier
@description: 
"""


import numpy as np
import matplotlib.pyplot as plt
import sklearn, time
import config
import itertools

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import linear_model

X_train = np.loadtxt(config.CSV_configuration.CSV_DICT['X_train'], delimiter=',')
y_train = np.loadtxt(config.CSV_configuration.CSV_DICT['y_train'] ,delimiter=',')
X_test = np.loadtxt(config.CSV_configuration.CSV_DICT['X_test'], delimiter=',')
y_test = np.loadtxt(config.CSV_configuration.CSV_DICT['y_test'] ,delimiter=',')


X_train =  sklearn.preprocessing.scale(X_train)
X_test =  sklearn.preprocessing.scale(X_test)
#scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))

#scaler = sklearn.preprocessing.StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.fit_transform(X_test)

print(X_train.shape)
print(X_test.shape)

GENRES=['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']


def confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
   
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_cnf(model,dataset_x,dataset_y,GENRES):
    true_y=dataset_y
    true_x=dataset_x
    pred=model.predict(true_x)

    
    print("Real Test dataset labels: \n{}\n".format(true_y))
    print("Predicted Test dataset labels: \n{}".format(pred))

    cnf_matrix=sklearn.metrics.confusion_matrix(true_y,pred)
    plt.figure()
    confusion_matrix(cnf_matrix,classes=GENRES,title='Confusion matrix')
    plt.show()



start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
print("Training Score: {:.3f}".format(knn.score(X_train,y_train)))
print("Test score: {:.3f}".format(knn.score(X_test,y_test)))  

plot_cnf(knn,X_test,y_test,GENRES)
print("---- %s Seconds -----" % (time.time() - start_time))





results_knn = []

start_time = time.time()
for i in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    results_knn.append(knn.score(X_test,y_test))
    
max_accuracy_knn = max(results_knn)
best_k = 1+results_knn.index(max(results_knn))
print("Max Accuracy is {:.3f} on test dataset with {} neighbors.\n".format(max_accuracy_knn,best_k))

plt.plot(np.arange(1,11),results_knn)
plt.xlabel("n Neighbors")
plt.ylabel("Accuracy")
plt.show()

print("---- %s Seconds -----" % (time.time() - start_time))



start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train,y_train)
print("Training Score: {:.3f}".format(knn.score(X_train,y_train)))
print("Test score: {:.3f}".format(knn.score(X_test,y_test)))  

plot_cnf(knn,X_test,y_test,GENRES)
print("---- %s Seconds -----" % (time.time() - start_time))



