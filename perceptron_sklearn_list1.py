import sklearn.metrics as metric
from sklearn.linear_model import Perceptron

# training data

training_data = [[1, 1, 1, 0, 1], [1, 0, 0, 1, 0], [1, 1, 1, 0, 0], [1, 1, 0, 1, 1], [1, 1, 0, 0, 1], [1, 0, 0, 1, 1]]
y_desired = [0, 1, 1, 0, 1, 0]

# test data
testing_data = [[1, 0, 0, 0, 1], [1, 1, 1, 1, 1]]

# Network to train, predict and measure the accuracy of your prediction.
perceptron = Perceptron(max_iter=5000, eta0=0.4)  # set the method
perceptron.coef_ = [0.1, 0.2, -0.2, -0.2, -0.3]
perceptron.fit(training_data, y_desired)  # training

y_prediction = perceptron.predict(training_data)  # prediction with training data
print(f'desired:\t{y_desired}')
print(f'predicted:\t{y_prediction}\n')

accuracy = metric.accuracy_score(y_desired, y_prediction, normalize=True)
print(f'accuracy:\t{accuracy}')
print(f"Perceptron's accuracy score\t: {accuracy}")

# show the synapsis weights w0, w1, w2, ...
print(f'Weights after training: {perceptron.intercept_}, {perceptron.coef_}')

y_testing = perceptron.predict(testing_data)  # prediction with test data
print("y_test = ", y_testing)  # show the output
