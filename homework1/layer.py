from sigmoid import SigmoidActivation
from softmax import SoftmaxActivation
import numpy as np


class Layer:

    def __init__(self, activation: str, number_of_units: int, input_size: int):
        """
        Initializes a layer in a neural network with a specified activation function, number of units, and input size.

        :param activation: The activation function for the layer ('sigmoid' or 'softmax').
        :param number_of_units: The number of units (neurons) in the layer.
        :param input_size: The size of the input to the layer.
        """

        # Initialize the activation function based on the provided string
        if activation == 'sigmoid':
            self.activation = SigmoidActivation()
        elif activation == 'softmax':
            self.activation = SoftmaxActivation()
        else:
            self.activation = None

        # Initialize weights and bias
        self.weights_matrix = np.random.normal(0.0, 0.2, size=(input_size, number_of_units))
        self.bias = np.zeros(number_of_units)

    def forward(self, input_array, di_list):
        """
        Performs the forward pass through the layer.

        :param input_array: The input array to the layer.
        :param di_list: List to store activations during forward pass.
        :return: The output of the layer after the forward pass.
        """

        pre_bias = np.matmul(input_array, self.weights_matrix)  # calculating the preactivation before bias
        preactivation = pre_bias + self.bias  # adding bias to preactivation
        activation = self.activation(preactivation)
        di_list.append(activation)  # Store the activation for potential later use
        return activation

    def weight_backwards(self, error_signal, activation):
        """
        Updates the weights based on the computed weight gradient.

        :param error_signal: The error signal from the next layer.
        :param activation: The activation from the current layer.
        """

        weight_gradient = np.dot(error_signal.T, activation)  # Calculation of the weight gradient
        # calculation of the mean gradient over the elements if the batch for weight updates
        self.weights_matrix = self.weights_matrix + 0.001 * weight_gradient.T  # updating weights

    def bias_backward(self, error_signal):
        """
        Updates the bias based on the computed bias gradient.

        :param error_signal: The error signal from the next layer.
        """

        self.bias = self.bias + 0.001 * np.mean(error_signal, axis=0)

    def backward(self, activations, error_signal, index):
        """
        Performs the backward pass through the layer to update parameters.

        :param activations: List of layer activations during forward pass.
        :param error_signal: The error signal from the next layer.
        :param index: The index of the current layer in the network.
        :return: The computed error signal for the current layer.
        """

        # Backward pass through the activation function
        if isinstance(self.activation, SigmoidActivation):
            error_signal = self.activation.backward(activations[index], error_signal)

        # safe weights before update to use them for calculating error signal for next layer
        weights = self.weights_matrix

        # Update bias and weights based on the computed gradients
        self.bias_backward(error_signal)
        self.weight_backwards(error_signal, activations[index - 1])

        # Compute and return the error signal for the next layer
        return error_signal.dot(weights.T)
