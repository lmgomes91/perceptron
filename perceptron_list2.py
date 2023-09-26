
from numpy import ndarray
import sklearn.metrics as metric
from sklearn.linear_model import Perceptron
import numpy as np
from sklearn.preprocessing import StandardScaler


def sklearn_perceptron(
        training_data: ndarray,
        y_desired: ndarray,
        testing_data: ndarray,
        learning_rate: float,
        weights: ndarray,
        cycles: int
) -> None:
    # Network to train, predict and measure the accuracy of your prediction.
    if cycles:
        perceptron = Perceptron(max_iter=cycles, eta0=learning_rate, tol=0.019)  # set the method
    else:
        perceptron = Perceptron(eta0=learning_rate, tol=0.019)  # set the method
    if weights.size:
        perceptron.coef_ = weights # noqa
    perceptron.fit(training_data, y_desired)  # training

    y_prediction = perceptron.predict(training_data)  # prediction with training data
    print(f'desired:\t{y_desired}')
    print(f'predicted:\t{y_prediction}\n')

    accuracy = metric.accuracy_score(y_desired, y_prediction, normalize=True)
    print(f'accuracy:\t{accuracy}')
    print(f"Perceptron's accuracy score\t: {accuracy}")

    # show synapse weights w0, w1, w2, ..., wn
    print(f'Weights after training: {perceptron.intercept_}, {perceptron.coef_}')

    y_testing = perceptron.predict(testing_data)  # prediction with test data
    print("y_test = ", y_testing)  # show the output


def exercise_1() -> None:
    training_data = np.array([
        [1, 1, 1, 1, 0, 1, 0, 0, 1, 0],  # T
        [1, 1, 0, 1, 1, 1, 1, 1, 0, 1],  # H
        [1, 0, 1, 0, 0, 1, 0, 0, 1, 0],  # I
        [1, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # x
        [1, 1, 1, 1, 0, 1, 0, 1, 1, 1],  # Z
        [1, 1, 0, 1, 0, 1, 0, 0, 1, 0],  # Y
        [1, 1, 0, 0, 1, 0, 0, 1, 1, 1],  # L
        [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],  # O
        [1, 1, 0, 1, 1, 0, 1, 1, 1, 1],  # U
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # all black
    ])
    weights = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y_desired = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    learning_rate = 0.01
    testing_data = np.array([
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 0],  # T modified
            [1, 1, 0, 0, 1, 1, 1, 1, 0, 1]  # H modified
    ])

    sklearn_perceptron(training_data, y_desired, testing_data, learning_rate, weights, 0)


def exercise_2() -> None:
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
    X_train_scaled = scaler.transform(X)
    # Aply the SAME scaler to the X test data
    X_test_scaled = scaler.transform(X_testing)
    # The following code is the example of how you will use Perceptron Neura
    # Network to train, predict and measure the accuracy of your prediction.
    ptn = Perceptron(max_iter=5000, eta0=0.01, tol=0.019)  # set the method
    ptn.fit(X_train_scaled, yd)  # training
    y_pred = ptn.predict(X_train_scaled) # prediction with training data
    print("yd = ", yd)  # show the desired output
    print("y = ", y_pred)  # show the output
    accuracy = metric.accuracy_score(yd, y_pred, normalize=True)
    print('acuracy = ', accuracy)
    print("Perceptron's accuracy score: {}".format(accuracy))
    # show the synapsis weights w0, w1, w2, ...
    print('Pesos apos treinamento (w0, w1, w2): ', ptn.intercept_, ptn.coef_)
    y_teste = ptn.predict(X_test_scaled)  # prediction with test data
    print("yteste = ", y_teste)  # show the output
    # Obtaining f(x) scores.
    pred_scores = ptn.decision_function(X_train_scaled)
    print("Perceptron's Children scores: {}".format(pred_scores))


def exercise_3() -> None:
    training_data = np.array([
        [1, 0, 0, 1],  # chicken
        [1, 1, 0, 1],  # elephant
        [1, 0, 0, 0],  # fish
        [1, 1, 1, 0],  # platypus
        [1, 0, 1, 1],  # scorpion
        [1, 1, 0, 0]  # whale
    ])
    weights = np.array([])
    y_desired = np.array([0, 1, 0, 1, 0, 1])
    learning_rate = 0.01
    testing_data = np.array([
        [1, 1, 1, 1],  # Echidna
        [1, 0, 1, 0]  # anaconda
    ])
    cycles = 2

    sklearn_perceptron(training_data, y_desired, testing_data, learning_rate, weights, cycles)


def main():
    # exercise_1()
    exercise_2()
    # exercise_3()


if __name__ == '__main__':
    main()
