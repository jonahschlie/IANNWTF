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
        # print(self.weights_bias_matrix)
        self.bias = np.zeros(number_of_units)

    def forward(self, input_array):
        pre_bias = np.matmul(input_array, self.weights_matrix)
        preactivation = pre_bias + self.bias
        return self.activation((preactivation))
