import numpy as np


class SigmoidActivation:

    def __call__(self, preactivation_matrix):
        activations = np.empty(shape=preactivation_matrix.shape)

        for row_idx, _ in enumerate(preactivation_matrix):
            for col_idx, _ in enumerate(preactivation_matrix[0]):
                activations[row_idx][col_idx] = 1 / (1 + np.exp(preactivation_matrix[row_idx][col_idx]))

        # return activation
        return activations
        # return 1 / (1 + np.exp(-preactivation_matrix))

    def backward(self, activation, error_signal):
        sigmoid_derivative = activation * (1 - activation)
        error_signal = error_signal * sigmoid_derivative
        return error_signal
