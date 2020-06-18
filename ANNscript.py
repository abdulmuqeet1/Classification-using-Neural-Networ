# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 23:12:59 2020

@author: Abdul

solution of ANN excercise

"""

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

## data preprocessing
# - read data  - label encoding  - split data  - feature scalling  - building ANN  - predicting result and comparing

dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:,3:13].values
y = dataset.iloc[:,-1].values

# encoding categorical data
labelencoder1 = LabelEncoder()
x[:, 1] = labelencoder1.fit_transform(x[:, 1])
labelencoder2 = LabelEncoder()
x[:, 2] = labelencoder2.fit_transform(x[:, 2])

#applying onehotencoder -- not running to check the performance without it
a = make_column_transformer(
    (OneHotEncoder(categories='auto'), [1]),
    remainder='passthrough')
x=a.fit_transform(x)

# splitting the dataset into the training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)

# feature scalling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# building ANN
classifier = Sequential()

#adding input layers with dropout
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=10))
classifier.add(Dropout(p=0.1))

#adding hidden layers
classifier.add(Dense(output_dim=6, init='uniform', activation='sigmoid'))
classifier.add(Dropout(p=0.1))

#adding another hidden layers
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

# compilinf the ANN
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])

# fitting ANN to the training set 
classifier.fit(x_train, y_train, batch_size=1, nb_epoch=100)

#predicting the result
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

 
# building the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# making array of new customer
new_prediction = classifier.predict(sc.transform(np.array([[0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]] )))
new_prediction = (new_prediction > 0.5)

'''
new cussomer info 
geography : france
credit score : 600
gender  male
age  40
 tenure  3
balance  60000
number of product  2
has credit card  yes
is active member  yes 
estimated salary  50000
'''

''' part 4 -- evaluating, improving and tuning the ANN'''

# evaluating the ANN


def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=10))
    classifier.add(Dropout(p=0.1))
    classifier.add(Dense(output_dim=6, init='uniform', activation='sigmoid'))
    classifier.add(Dropout(p=0.1))
    
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch=100)
accuracies = cross_val_score(estimator = classifier, X= x_train, y=y_train, cv=10, n_jobs=-1)


mean = accuracies.mean()
variance =accuracies.std()
    
# improving the ANN
# intoducing the dropout method to deal with overfitting


# tuning the ANN
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=10))
    classifier.add(Dense(output_dim=6, init='uniform', activation='sigmoid'))
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    classifier.compile(optimizer = optimizer, loss='binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size':[25,32], 'nb_epoch':[100,250],'optimizer':['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator= classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)

grid_search.fit(x_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

