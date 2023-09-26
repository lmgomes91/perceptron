# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 11:35:08 2019

@author: nilto
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

plt.style.use('seaborn-poster')
# %matplotlib inline
'exec(%matplotlib inline)'

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from timeit import default_timer as timer

# Read dataset to pandas dataframe
# data = pd.read_csv("../ProgPythonAnaconda/arq_teste_csv.csv")
data = pd.read_csv("arq_teste_csv.csv")

# Assign data from first four columns to X variable
x = data.iloc[:, 1:5]
y = data.iloc[:, 5:6]

x_training = x
y_training = y

initreino = timer()
mlp = MLPRegressor(hidden_layer_sizes=(10,), max_iter=2000, solver='lbfgs', \
                   alpha=0.01, activation='tanh', random_state=8)
# training
y_training = (np.array(y_training)).ravel()
# mlp.fit(x_training, y_training.ravel())
mlp.fit(x_training, y_training)
fimtreino = timer()

# ypred = mlp.predict(x_test[:, None])                 # prediction
initeste = timer()
ypred = mlp.predict(x_training)
fimteste = timer()

tempotreino = fimtreino - initreino
tempoteste = fimteste - initeste
tempototal = tempotreino + tempoteste

mse = mean_squared_error(y_training, ypred)
y_true = y_training

print("\nRESULTADOS")
print('\nMSE (%) = ', mse * 100)
print("\nTempo de treinamento = ", "%.4f" % tempotreino, "segundos")
print("Tempo de teste = ", "%.4f" % tempoteste, "segundos")
print("Tempo total (treinamento + teste) = ", "%.4f" % tempototal, "segundos")

tempotreinoms = tempotreino * 1000
tempotestems = tempoteste * 1000
tempototalms = tempototal * 1000

print("\nTempo de treinamento = ", "%.4f" % tempotreinoms, "ms")
print("Tempo de teste = ", "%.4f" % tempotestems, "ms")
print("Tempo total (treinamento + teste) = ", "%.4f" % tempototalms, "ms")

from matplotlib import pyplot as plt

plt.title('Aproximação de função Y')
plt.ylabel('Y')
plt.xlabel(u'X')
reg_val, = plt.plot(ypred, color='b', label=u'MLP Regression')
true_val, = plt.plot(y_true, color='g', label='True Values')
plt.xlim([0, 200])  # 85
plt.legend(handles=[true_val, reg_val])
plt.show()
