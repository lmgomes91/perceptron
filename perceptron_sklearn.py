# import numpy as np
# import matplotlib.pyplot as plt
import sklearn.metrics as metric
# from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

# X training data

X = [[100, 20], [100, 26], [100, 30], [100, 32], [102, 21], [105, 22], [107, 32], [110, 35], [111, 25], [114, 24],
     [116, 36], [118, 27]]  # Fat Children
yd = [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0]  # % 1 = fat, 0 = not fat
# X test data
X_testing = [[120, 32]]
# Creating a StandardScaler.
# This object normalizes features to zero mean and unit variance.
scaler = StandardScaler()
scaler.fit(X)

# Normalizing train and test data.
# Aply the scaler to the X training data
X_train_scaled = scaler.transform(X)
# Aply the SAME scaler to the X test data
X_test_scaled = scaler.transform(X_testing)
# The following code is the example of how you will use Perceptron Neural

# Network to train, predict and measure the accuracy of your prediction.
ptn = Perceptron(max_iter=5000, eta0=0.01, tol=0.019)  # set the method
ptn.fit(X_train_scaled, yd)  # training
y_pred = ptn.predict(X_train_scaled)  # prediction with training data

print("yd = ", yd)  # show the desired output
print("y = ", y_pred)  # show the output
accuracy = metric.accuracy_score(yd, y_pred, normalize=True)
print('acuracy = ', accuracy)
print("Perceptron's accuracy score: {}".format(accuracy))

# show the synapsis weights w0, w1, w2, ...

print('Pesos apos treinamento (w0, w1, w2): ', ptn.intercept_, ptn.coef_)

y_teste = ptn.predict(X_test_scaled)  # prediction with test data
print("yteste = ", y_teste)  # show the output
