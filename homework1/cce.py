import numpy as np


class CCE:

    def __call__(self, output, target):
        return -1 * np.sum(target * np.log(output), axis=1, keepdims=True)
        #return np.mean(-1 * np.sum(target * np.log(output), axis=1, keepdims=True))

    def backward(self, prediction, target):
        # dy = - np.sum((target / prediction), axis=1, keepdims=True)
        dy = - target / (prediction + 10**-1000)
        return dy



