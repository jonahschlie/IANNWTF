import numpy as np


class SigmoidActivation:

    def __call__(self, preactivation_matrix):
        """
        Computes the sigmoid activation for a given preactivation matrix.

        :param preactivation_matrix: The preactivation values before applying the sigmoid.
        :return: The sigmoid activations.
        """

        return 1 / (1 + np.exp(-preactivation_matrix))

    def backward(self, activation, error_signal):
        """
        Computes the backward pass for the sigmoid activation.

        :param activation: The activation from forward step values.
        :param error_signal: The error signal from the next layer.
        :return: The computed error signal for the current layer.
        """

        # Compute the derivative of the sigmoid with respect to the preactivation
        sigmoid_derivative = activation * (1 - activation)

        # Multiply the error signal by the sigmoid derivative
        error_signal = error_signal * sigmoid_derivative

        # Return the computed error signal for the current layer
        return error_signal
