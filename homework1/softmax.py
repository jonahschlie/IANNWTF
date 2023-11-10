import numpy as np


class SoftmaxActivation:

    def __call__(self, preactivation_matrix):
        activations = np.empty(shape=preactivation_matrix.shape)

        for row_idx, _ in enumerate(preactivation_matrix):
            for col_idx, _ in enumerate(preactivation_matrix[0]):
                activations[row_idx][col_idx] = np.exp(preactivation_matrix[row_idx][col_idx]) / \
                                                np.sum(np.exp(preactivation_matrix[row_idx]))
        # return
        return activations


        # e_x = np.exp(preactivation_matrix)
        # return e_x / np.sum(e_x, axis=1, keepdims=True)

    def backward(self, activation, dl):
        softmax_derivate = activation * (1-activation)
        dl = dl * softmax_derivate
        return dl

