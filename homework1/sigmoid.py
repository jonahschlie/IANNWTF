import numpy as np


class SigmoidActivation:

    def __call__(self, preactivation_matrix):
        return 1 / (1 + np.exp(-preactivation_matrix))
