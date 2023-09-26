# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:01:09 2019

@author: nilto
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neural_network import MLPClassifier  # neural network
from sklearn import metrics
from timeit import default_timer as timer

# Input data files are available in the "../ProgPythonAnaconda/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will
# list the files in the ProgPythonAnaconda directory
# import os
# print(os.listdir("../ProgPythonAnaconda"))

# Read dataset to pandas dataframe
data = pd.read_csv("iris.csv")

# Preprocessing
# You can see that our dataset has five columns.
# The task is to predict the class (which are the values in the fifth column) that the iris plant
# belongs to, which is based upon the sepal-length, sepal-width, petal-length and petal-width
# (the first four columns). The next step is to split our dataset into attributes and labels.
# Execute the following script to do so:

# Assign data from first four columns to X variable
X = data.iloc[:, 0:4]

# Assign data from first fifth columns to y variable
y = data.select_dtypes(include=[object])

# We have three unique classes 'Iris-setosa', 'Iris-versicolor' and 'Iris-virginica'.
# Let's convert these categorical values to numerical values.
# To do so we will use Scikit-Learn's LabelEncoder class. Execute the following script:

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
yd = y.apply(le.fit_transform)

# Train Test Split
# To avoid over-fitting, we will divide our dataset into training and test splits.
# The training data will be used to train the neural network and the test data will be used
# to evaluate the performance of the neural network. .
# To create training and test splits, execute the following script:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, yd, test_size=0.30)

# The above script splits 80% of the dataset into our training set and the other 30% in to test data.

# Feature Scaling
# Before making actual predictions, it is always a good practice to scale the features so
# that all of them can be uniformly evaluated.
# The following script performs feature scaling:

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Training and Predictions

# And now it's finally time to do what you have been waiting for, train a neural network that
# can actually make predictions. To do this, execute the following script:
# from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(20, 20,), max_iter=10000, activation='logistic', solver='sgd',
                    learning_rate='constant', verbose=True, tol=1e-6, learning_rate_init=.1)

# mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, activation='relu', solver='sgd', learning_rate='constant', verbose=True, tol=1e-6, learning_rate_init=.1)
# mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, activation='logistic', solver='adam', learning_rate='constant', verbose=True, tol=1e-6, learning_rate_init=.1)
# mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, activation='relu', solver='adam', learning_rate='constant', verbose=True, tol=1e-6, learning_rate_init=.1)


# learning_rate='adaptive'

initreino = timer()
mlp.fit(X_train, y_train.values.ravel())
fimtreino = timer()

# The final step is to make predictions on our test data. To do so, execute the following script:
initeste = timer()
predictions = mlp.predict(X_test)
fimteste = timer()

tempotreino = fimtreino - initreino
tempoteste = fimteste - initeste
tempototal = tempotreino + tempoteste

print('\nThe accuracy of the Multi-layer Perceptron is:', metrics.accuracy_score(predictions, y_test))
print('\nMatriz de Confusão')
# Evaluating the Algorithm
# We created our algorithm and we made some predictions on the test dataset.
# Now is the time to evaluate how well our algorithm performs. To evaluate an algorithm,
# the most commonly used metrics are a confusion matrix, precision, recall, and f1 score.
# The confusion_matrix and classification_report methods of the sklearn.metrics library
# can help us find these scores.
# The following script generates evaluation report for our algorithm:

target_names = ['setosa', 'versicolor', 'virginica']
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions, target_names=target_names))
# print(classification_report(y_test,predictions))

print("Tempo de treinamento = ", "%.4f" % tempotreino, "segundos")
print("Tempo de teste = ", "%.4f" % tempoteste, "segundos")
print("Tempo total (treinamento + teste) = ", "%.4f" % tempototal, "segundos")

tempotreinoms = tempotreino * 1000
tempotestems = tempoteste * 1000
tempototalms = tempototal * 1000
print("\nTempo de treinamento = ", "%.4f" % tempotreinoms, "ms")
print("Tempo de teste = ", "%.4f" % tempotestems, "ms")
print("Tempo total (treinamento + teste) = ", "%.4f" % tempototalms, "ms\n")
