import numpy as np


def calculate_net(pattern: np.ndarray, weights: np.ndarray) -> float:
    net = 0.0

    for p in range(pattern.size):
        net += pattern[0, p] * weights[p]

    return round(net, 2)


def evaluate_pattern_output(net: float) -> int:
    if net < 0:
        return 0
    else:
        return 1


def recalculate_weights(weights: np.ndarray, pattern: np.ndarray, error: int, n: float) -> np.ndarray:
    for w in range(weights.size):
        weights[w] = weights[w] + n * pattern[0, w] * error
    return weights


def perceptron_trainer(n: float, patterns: np.matrix, weights: np.ndarray, output: np.ndarray) -> np.ndarray:
    while True:
        cycle_result = np.full(patterns.shape[0], False, dtype=bool)
        for p in range(patterns.shape[0]):
            net = calculate_net(patterns[p], weights)
            pattern_output = evaluate_pattern_output(net)

            if not pattern_output == int(output[p]):
                weights = recalculate_weights(weights, patterns[p], output[p] - pattern_output, n)
            else:
                cycle_result[p] = True

        print(cycle_result)
        if np.all(cycle_result == True): # noqa
            break

    return weights


def perceptron_neural_network(patterns: np.matrix, weights: np.ndarray) -> np.ndarray:
    output = np.full(patterns.shape[0], -1, dtype=int)
    for p in range(patterns.shape[0]):
        net = calculate_net(patterns[p], weights)
        output[p] = evaluate_pattern_output(net)
    return output


def main():
    n = 0.4
    weights = np.array([-0.5, 0.4, -0.6, 0.6])
    trainer_patterns = np.matrix([[1, 0, 0, 1], [1, 1, 1, 0]])
    desired_output = np.array([0, 1])

    weights = perceptron_trainer(n, trainer_patterns, weights, desired_output)
    print(f'\nweights after training: {weights}\n')

    patterns = np.matrix([[1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 1]])
    print(f'Output from pattern: {perceptron_neural_network(patterns, weights)}')


if __name__ == '__main__':
    main()
