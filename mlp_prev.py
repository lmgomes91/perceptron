from sklearn.neural_network import MLPRegressor  # linear regression model.

X_training = [
    [0.3, 0.1, 0.1],
    [0.03, 0.02, 0],
    [1, 1, 1],
    [0.4, 0.15, 1],
    [0.9, 0.8, 0.8],
    [0.5, 0.5, 0.9]
]
y_training = [0.19, 0.11, 0.6, 0.31, 0.52, 0.39]

X_testing = [
    [0.70, 0.60, 0.85]
]

mlp = MLPRegressor(
    hidden_layer_sizes=(15,),
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    max_iter=1000,
    learning_rate_init=0.01,
    alpha=0.01,
    verbose=False
)
mlp.fit(X_training, y_training)  # training
y_pred = mlp.predict(X_testing)  # prediction
print("\npredictions:", y_pred)
