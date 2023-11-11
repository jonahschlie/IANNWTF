from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import random
from layer import Layer
from mlp import MLP
from cce import CCE

digits = load_digits()


def one_hot_encoding(target_number):
    """
    This function performs one-hot encoding for a target number.

    :param target_number: The target number to be one-hot encoded.
    :return: A one-hot encoded array representing the target number.
    """

    encoded = np.zeros(10)  # initialize empty Numpy array of size 10
    encoded[target_number] = 1  # set value in array at target position to one
    return encoded


def shuffle_data(batch_size, tuples_array):
    """
    This generator function shuffles the input-target tuples and yields minibatches.

    :param batch_size: Size of one minibatch.
    :param tuples_array: List of tuples containing input-target pairs.

    :return: A minibatch-generator object that yields shuffled minibatches.
    """

    random.shuffle(tuples_array)  # Shuffle the input-target tuples randomly
    num_samples = len(tuples_array)  # Get the total number of samples in the dataset
    num_minibatches = num_samples // batch_size  # Calculate the number of minibatches

    # Iterate over each minibatch
    for minibatch_index in range(num_minibatches):
        start_index = minibatch_index * batch_size
        end_index = (minibatch_index + 1) * batch_size

        minibatch_data = tuples_array[start_index:end_index]  # Extract the current minibatch data
        inputs, targets = zip(*minibatch_data)  # Unzip the input and target values from the minibatch data

        # Convert the inputs and targets into Numpy arrays
        input_batch = np.array(inputs)
        target_batch = np.array(targets)

        yield input_batch, target_batch


# generating input-target-tuples from training data
target_input_tuples = [(2 * (digits.data[i].astype(np.float32) / 16) -
                        1, one_hot_encoding(digits.target[i])) for i in range(len(digits.data))]

minibatch_generator_object = shuffle_data(1, target_input_tuples)

mlp_object = MLP(2, [64, 10])  # initializing the MLP object
cce_object = CCE()  # initializing the CCE object


def train(mlp, minibatch_generator, epochs, cce):
    """
    This function trains a multi-layer perceptron (MLP) using a provided minibatch generator and a specified loss function.

    :param mlp: The MLP object to be trained.
    :param minibatch_generator: The generator object providing minibatches of input and target data.
    :param epochs: The number of training epochs.
    :param cce: The categorical cross-entropy loss function.
    """

    for i in range(0, epochs):
        for input_batch, target_batch in minibatch_generator:
            output = mlp.forward(input_batch)  # passing the input data through the network
            loss = cce(output, target_batch)  # calculating the loss value
            print(loss)
            dcce = cce.backward(output, target_batch)
            mlp.backward(dcce)  # updating the networks weights and biases through backpropagation


train(mlp_object, minibatch_generator_object, 1, cce_object)

'''

input_batch, target_batch = minibatch_generator.__next__()

output = mlp.forward(input_batch)
losses = cce(output, target_batch)
dcce = cce.backward(output,target_batch)
mlp.backward(dcce)

print("==============Neue Runde===============")

input_batch, target_batch = minibatch_generator.__next__()

output = mlp.forward(input_batch)
losses = cce(output, target_batch)
dcce = cce.backward(output,target_batch)
mlp.backward(dcce)

input_batch, target_batch = minibatch_generator.__next__()

output = mlp.forward(input_batch)
losses = cce(output, target_batch)
print(losses)
dcce = cce.backward(output,target_batch)
print(dcce)
mlp.backward(dcce)

output = mlp.forward(input)

input_batch, target_batch = minibatch_generator.__next__()

output = mlp.forward(input_batch)
# print(mlp.di_list)
losses = cce(output, target_batch)
dcce = cce.backward(output,target_batch)
mlp.backward(dcce)

error = cce(output, target)
dcce = cce.backward(output, target)
dsoftmax = output * (1-output)
error_signal = dsoftmax * dcce

print(error)
print(dcce)
print(dsoftmax)
print(error_signal)

for layer in mlp.layers:
    print(layer.weights_matrix.shape)
    
plt.gray()
plt.matshow(target_input_tuples[33][0])
plt.show()
'''
