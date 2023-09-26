from sklearn.neural_network import MLPClassifier


X_training = [
    [1, 1, 0, 1],
    [0, 0, 1, 0],
    [1, 1, 0, 0],
    [1, 0, 1, 1],
    [1, 0, 0, 1],
    [0, 0, 1, 1]
]
y_training = [0, 1, 1, 0, 1, 0]

X_testing = [
    [0, 0, 0, 1],
    [1, 1, 1, 1],
    [0, 0, 0, 0]
]

mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=1000,
    solver='sgd',
    verbose=False,
    tol=1e-4,
    learning_rate_init=.1
)
print(mlp.fit(X_training, y_training))  # training
y_pred = mlp.predict(X_testing)  # noqa prediction

print('\npredictions:', y_pred)  # show the output

