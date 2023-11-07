import numpy as np


class CCE:

    def __call__(self, output, target):

        return -1 * np.sum(target * np.log(output), axis=1, keepdims=True)
