import numpy as np


class CCE:

    def __call__(self, output, target):
        # return -1 * np.sum(target * np.log(output), axis=1, keepdims=True)
        return np.mean(-1 * np.sum(target * np.log(output + 1e-10), axis=1, keepdims=True))

    def backward(self, prediction, target):
        dy = target - prediction
        return dy



