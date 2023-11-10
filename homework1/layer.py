from sigmoid import SigmoidActivation
from softmax import SoftmaxActivation
import numpy as np


class Layer:

    def __init__(self, activation: str, number_of_units: int, input_size: int):
        if activation == 'sigmoid':
            self.activation = SigmoidActivation()
        elif activation == 'softmax':
            self.activation = SoftmaxActivation()
        else:
            self.activation = None
        self.weights_matrix = np.random.normal(0.0, 0.2, size=(input_size, number_of_units))
        self.bias = np.zeros(number_of_units)

    def forward(self, input_array, di_list):
        pre_bias = np.matmul(input_array, self.weights_matrix)
        preactivation = pre_bias + self.bias
        activation = self.activation(preactivation)
        di_list.append(activation)
        return activation

    def weight_backwards(self, error_signal, activation):
        weight_gradient = np.matmul(error_signal.T, activation)
        mean_gradient = np.mean(weight_gradient.T, axis=0)
        self.weights_matrix = self.weights_matrix - 0.01 * mean_gradient

    def bias_backward(self, error_signal):
        self.bias = self.bias - 0.01 * np.mean(error_signal, axis=0)

    def backward(self, activations, error_signal, index):
        error_signal = self.activation.backward(activations[index], error_signal)
        weights = self.weights_matrix
        self.bias_backward(error_signal)
        self.weight_backwards(error_signal, activations[index - 1])
        return np.matmul(error_signal, weights.T)
