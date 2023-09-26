from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

x = [[100, 20], [100, 26], [100, 30], [100, 32], [102, 21], [105, 22], [107, 32], [110,
                                                                                   35], [111, 25], [114, 24], [116, 36],
     [118, 27]]  # Fat Children
yd = [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0]  # 1 = fat, 0 = not fat
x = pd.DataFrame(x, columns=['x1', 'x2'])
print("\nNormal_x")

print(x)
scaler = MinMaxScaler()
scaled_x = scaler.fit_transform(x)
scaled_x = pd.DataFrame(scaled_x, columns=['x1', 'x2'])
print("\nscaled_x")
print(scaled_x)
X_training = scaled_x
y_training = yd
X_testing = X_training
y_test = y_training
print('\nSaida desejada')
print(y_training)
mlp = MLPClassifier(hidden_layer_sizes=(4,), max_iter=20, solver='sgd',
                    verbose=True, tol=1e-4, learning_rate_init=.1)
mlp.fit(X_training, y_training)  # training
y_pred = mlp.predict(X_testing)  # prediction
print('\nSaida obtida:')
print(y_pred)  # show the output
accuracy = accuracy_score(y_test, y_pred)

print("MLP's accuracy score: {}".format(accuracy))
print('\nConfusion matrix')
print(confusion_matrix(y_training, y_pred))
print('\nClassification report')
print(classification_report(y_training, y_pred))
