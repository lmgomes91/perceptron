from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

X_training = [[1, 1], [1, 0], [0, 1], [0, 0]]
y_training = [0, 1, 1, 0]
X_testing = X_training
y_test = y_training

mlp = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000, solver='sgd', verbose=True,
                    tol=1e-4, learning_rate_init=.1)
mlp.fit(X_training, y_training)  # training

y_pred = mlp.predict(X_testing)  # noqa prediction

print('\npredictions:', y_pred)  # show the output
print('expected:', y_training)
accuracy = accuracy_score(y_test, y_pred)
print("\nMLP's accuracy score: {}".format(accuracy))
print('\nConfusion matrix:\n {}'.format(confusion_matrix(y_test, y_pred)))
print('\nClassification Report:\n {}'.format(classification_report(y_test, y_pred)))
