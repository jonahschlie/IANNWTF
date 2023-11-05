import numpy as np


class SoftmaxActivation:

    def call(self, preactivation_matrix):
        e_x = np.exp(preactivation_matrix)
        return e_x / np.sum(e_x, axis=1, keepdims=True)
