import numpy as np


class SoftmaxActivation:

    def __call__(self, preactivation_matrix):
        """
        Computes the softmax activation for a given preactivation matrix.

        :param preactivation_matrix: The preactivation values before applying the softmax.
        :return: The softmax activations.
        """

        exp_x = np.exp(preactivation_matrix - np.max(preactivation_matrix, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
