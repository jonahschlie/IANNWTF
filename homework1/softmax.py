import numpy as np


class SoftmaxActivation:

    def __call__(self, preactivation_matrix):
        """
        Computes the softmax activation for a given preactivation matrix.

        :param preactivation_matrix: The preactivation values before applying the softmax.
        :return: The softmax activations.
        """

        activations = np.empty(shape=preactivation_matrix.shape)

        # Iterate over each row of the preactivation matrix
        for row_idx, _ in enumerate(preactivation_matrix):
            # Compute the softmax for each element in the row
            for col_idx, _ in enumerate(preactivation_matrix[0]):
                activations[row_idx][col_idx] = np.exp(preactivation_matrix[row_idx][col_idx]) / \
                                                np.sum(np.exp(preactivation_matrix[row_idx]))

        # Return the computed softmax activations
        return activations


        # e_x = np.exp(preactivation_matrix)
        # return e_x / np.sum(e_x, axis=1, keepdims=True)

    def backward(self, activation, dl):
        """
        Computes the backward pass for the softmax activation.

        :param activation: The softmax activation values.
        :param dl: The error signal from the next layer.
        :return: The computed error signal for the current layer.
        """

        # Compute the derivative of softmax with respect to the preactivation
        softmax_derivative = activation * (1 - activation)

        # Multiply the error signal by the softmax derivative
        dl = dl * softmax_derivative

        # Return the computed error signal for the current layer
        return dl
