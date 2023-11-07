from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import random
from layer import Layer
from mlp import MLP
from cce import CCE

digits = load_digits()


def one_hot_encoding(target):
    encoded = np.zeros(10)
    encoded[target] = 1
    return encoded


def shuffle_data(batch_size, tuples_array):
    random.shuffle(tuples_array)
    num_samples = len(tuples_array)
    num_minibatches = num_samples // batch_size

    for minibatch_index in range(num_minibatches):
        start_index = minibatch_index * batch_size
        end_index = (minibatch_index + 1) * batch_size
        minibatch_data = tuples_array[start_index:end_index]

        inputs, targets = zip(*minibatch_data)
        input_batch = np.array(inputs)
        target_batch = np.array(targets)

        yield input_batch, target_batch


target_input_tuples = [(2 * (digits.data[i].astype(np.float32) / 16) - 1, one_hot_encoding(digits.target[i])) for i in
                       range(len(digits.data))]

minibatch_generator = shuffle_data(10, target_input_tuples)

input, target = minibatch_generator.__next__()

mlp = MLP(3, [64, 10, 10])

output = mlp.forward(input)

cce = CCE()

error = cce(output, target)

print(error)


'''

for layer in mlp.layers:
    print(layer.weights_matrix.shape)

# Das hier verwenden um durch die batches durchzuiterieren und das Training durchzuführen.
for i in range(0, 1):
    inputs, targets = minibatch_generator.__next__()
    print(inputs[5])
    print(targets[5])
    plt.gray()
    plt.matshow(inputs[5].reshape(8, 8))
    plt.show()
    
for i in range(0, 1):
    inputs, targets = minibatch_generator.__next__()
    results = layer.forward(inputs)
    print(results)
    print(np.sum(results[9]))


print(len(digits.images))
print(len(digits.data))
print(len(digits.target))

print(digits.images[1796])
print(digits.data[0])
print(digits.target[0])
print(digits.images[0].reshape(64))



plt.gray()
plt.matshow(target_input_tuples[33][0])
plt.show()
'''
