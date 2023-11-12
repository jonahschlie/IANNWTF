from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import random
from layer import Layer
from mlp import MLP
from cce import CCE

DIGITS = load_digits()
LOSS_PER_EPOCH = []
MINIBATCHSIZE = 64
NUM_EPOCHS = 300
ACCURACY_PER_EPOCH = []


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

def calc_accuracy(target_label, predicted_label):
    return np.equal(np.argmax(target_label, axis=-1), np.argmax(predicted_label, axis=-1))

def train(mlp, target_input_tuples, epochs, cce):
    """
    This function trains a multi-layer perceptron (MLP) using a provided minibatch generator and a specified loss function.

    :param mlp: The MLP object to be trained.
    :param minibatch_generator: The generator object providing minibatches of input and target data.
    :param epochs: The number of training epochs.
    :param cce: The categorical cross-entropy loss function.
    """
    num_samples = len(target_input_tuples)  # Get the total number of samples in the dataset
    num_minibatches = num_samples // MINIBATCHSIZE  # Calculate the number of minibatches
    
    loss_per_batch = dict.fromkeys(list(range(1, num_minibatches+1)))
    for i in range(0, epochs):
        print(f"Epoch {i}")
        minibatch_generator = shuffle_data(MINIBATCHSIZE, target_input_tuples)
        current_minibatch=1
        truePred = 0
        falsePred = 0
        for input_batch, target_batch in minibatch_generator:
            output = mlp.forward(input_batch)  # passing the input data through the network
            
            falsePred += np.size(calc_accuracy(target_batch, output)) - np.count_nonzero(calc_accuracy(target_batch, output))
            truePred += np.count_nonzero(calc_accuracy(target_batch, output))
            
            loss = cce(output, target_batch)  # calculating the loss value
            #print(loss)
            loss_per_batch[current_minibatch] = loss
            dcce = cce.backward(output, target_batch)
            mlp.backward(dcce)  # updating the networks weights and biases through backpropagation
            current_minibatch +=1
        ACCURACY_PER_EPOCH.append(truePred/(truePred+falsePred))
        LOSS_PER_EPOCH.append(sum(loss_per_batch.values())/len(loss_per_batch))

def main():
    np.random.seed(42)

    # generating input-target-tuples from training data
    target_input_tuples = [(DIGITS.data[i].astype(np.float32) / 16, one_hot_encoding(DIGITS.target[i])) for i in range(len(DIGITS.data))]

    mlp_object = MLP(2, [64, 10])  # initializing the MLP object
    cce_object = CCE()  # initializing the CCE object
    
    train(mlp_object, target_input_tuples, NUM_EPOCHS, cce_object)
    
    plt.plot(list(range(1, NUM_EPOCHS+1)), LOSS_PER_EPOCH)
    plt.xlabel("Epoch")
    plt.ylabel("Categorical Cross Entropy Loss")
    plt.show()
    
    plt.plot(list(range(1, NUM_EPOCHS+1)), ACCURACY_PER_EPOCH)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()



if __name__ == "__main__":
    main()