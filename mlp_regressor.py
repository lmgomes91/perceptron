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

np.random.seed(0)
x_training = 10 * np.random.rand(100)
plt.figure(figsize = (12,10))
y_training=2*x_training*x_training-0.21**x_training

plt.errorbar(x_training, y_training, 0.3, fmt='o')
xtest = np.linspace(0, 10, 1000)

mlp = MLPRegressor(hidden_layer_sizes=(10,), max_iter = 2000, solver='lbfgs', \
                   alpha=0.01, activation = 'tanh', random_state = 8)
ytest=2*xtest*xtest-0.21**xtest
mlp.fit(x_training[:, None], y_training)            # training
ypred = mlp.predict(xtest[:, None])                 # prediction

plt.figure(figsize = (12,10))
plt.errorbar(x_training, y_training, 0.3, fmt='o')
plt.plot(xtest, ypred, '-r', label = 'predicted', zorder = 10)
plt.plot(xtest, ytest, '-k', alpha=0.5, label = 'test model', zorder = 10)
plt.legend()
plt.show()
mse = mean_squared_error(ytest, ypred)
print("mse = ",mse)