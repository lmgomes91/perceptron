# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 21:28:46 2019

@author: nilto
"""

import numpy as np
import matplotlib.pyplot as plt
from RBFN import RBFN
from sklearn.metrics import mean_squared_error

# generating data
x = np.linspace(0, 10, 100)
yd = np.sin(x)

# fitting RBF-Network with data
model = RBFN(hidden_shape=30, sigma=1.)
model.fit(x, yd)
# xtrue = np.linspace(0, 10, 1000)
y_pred = model.predict(x)
# y_pred = model.predict(xtrue)

# plotting 1D interpolation
plt.plot(x, yd, 'b-', label='real')
plt.plot(x, y_pred, 'r-', label='fit')
plt.title('Interpolation using a RBFN')
# plt.text(10, 1.0, r'$\mu=100,\ \sigma=15$')
plt.ylabel('yd= sin(x)')
plt.xlabel('x')
plt.show()

mse = mean_squared_error(yd, y_pred)
print("mse = ", mse)

# Cria um gráfico dos valores reais, previsões da regressão linear e do modelo
# utilizando o último valor
# OPCIONAL - REQUER MATPLOTLIB
from matplotlib import pyplot as plt

plt.title('Interpolation using a RBFN')
plt.ylabel('yd= sin(x)')
plt.xlabel('x')
reg_val, = plt.plot(y_pred, color='r', label=u'RBF Interpolation')
true_val, = plt.plot(yd, color='b', label='True Values')
plt.xlim([0, 100])  # antes 85
plt.legend(handles=[true_val, reg_val])
plt.show()
