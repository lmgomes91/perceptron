from sklearn.neural_network import MLPClassifier
import pandas as pd

x = [[0, 0], [0, 1], [1, 0], [1, 1]]
yd = [[0, 0], [1, 0], [1, 0], [0, 1]]
X_training = x
y_training = yd
X_testing = X_training
y_test = y_training
mlp = MLPClassifier(hidden_layer_sizes=(20,), max_iter=1000, solver='sgd',
                    verbose=True, tol=1e-4, learning_rate_init=.1)
mlp.fit(X_training, y_training)  # training

y_pred = mlp.predict(X_testing)  # prediction
x = pd.DataFrame(x, columns=['x1', 'x2'])
print("\nEntradas:")
print(x)
yd = pd.DataFrame(yd, columns=['y1', 'y2'])
print("\nSaidas desejadas:")
print(yd)
y_pred = pd.DataFrame(y_pred, columns=['y1', 'y2'])
print("\nSaidas obtidas:")
print(y_pred)  # show the output
