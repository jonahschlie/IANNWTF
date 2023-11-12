from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import random
from layer import Layer
from mlp import MLP
from cce import CCE
import plotly
from plotly.graph_objects import *


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

def calc_accuracy(target_label: np.array, predicted_label: np.array) -> np.array:
    """Calculate which predicted outputs match the targets.

    Args:
        target_label (np.array): Numpy array with the correct one-hot encoded labels.
        predicted_label (np.array): Numpy array with the predicted probabilities of the classes.

    Returns:
        np.array: Numpy array of booleans corresponsing to whether predictions were correct.
    """
    # get class from arrays based on maximum probability
    target_class = np.argmax(target_label, axis=-1)
    predicted_class = np.argmax(predicted_label, axis=-1)

    # check which predicted class matches the target class
    return np.equal(target_class, predicted_class)

def train(mlp, target_input_tuples, epochs, cce):
    """
    This function trains a multi-layer perceptron (MLP) using a provided minibatch generator and a specified loss function.

    :param mlp: The MLP object to be trained.
    :param target_input_tuples: Tuples with input and class.
    :param epochs: The number of training epochs.
    :param cce: The categorical cross-entropy loss function.
    """
    num_samples = len(target_input_tuples)  # Get the total number of samples in the dataset
    num_minibatches = num_samples // MINIBATCHSIZE  # Calculate the number of minibatches
    
    loss_per_batch = dict.fromkeys(list(range(1, num_minibatches+1))) # create empty dict with the minibatch number (starting from 1) as keys
    
    for i in range(0, epochs):
        print(f"Epoch {i}")
        # generate new minibatch for each epoch
        minibatch_generator = shuffle_data(MINIBATCHSIZE, target_input_tuples)
        current_minibatch=1
        # number of true and false predictions each epoch
        truePred = 0
        falsePred = 0
        for input_batch, target_batch in minibatch_generator:
            output = mlp.forward(input_batch)  # passing the input data through the network
            
            # truePred is the number of all predictions (np.count_nonzero is faster than np.sum); result is added to value for whole epoch
            # falsePred is the number of all predictions - correct predictions; result is added to value for whole epoch
            truePred += np.count_nonzero(calc_accuracy(target_batch, output))
            falsePred += np.size(calc_accuracy(target_batch, output)) - np.count_nonzero(calc_accuracy(target_batch, output))
            
            loss = cce(output, target_batch)  # calculating the loss value
            #print(loss)
            loss_per_batch[current_minibatch] = loss # set loss for each batch in the dict
            dcce = cce.backward(output, target_batch)
            mlp.backward(dcce)  # updating the networks weights and biases through backpropagation
            current_minibatch +=1 # count up the current minibatch
        # append the accuracy and the average loss of the whole epoch over all minibatches
        ACCURACY_PER_EPOCH.append(truePred/(truePred+falsePred))
        LOSS_PER_EPOCH.append(sum(loss_per_batch.values())/len(loss_per_batch))

def main():
    # generating input-target-tuples from training data
    target_input_tuples = [(DIGITS.data[i].astype(np.float32) / 16, one_hot_encoding(DIGITS.target[i])) for i in range(len(DIGITS.data))]

    mlp_object = MLP(2, [64, 10])  # initializing the MLP object
    cce_object = CCE()  # initializing the CCE object
    
    train(mlp_object, target_input_tuples, NUM_EPOCHS, cce_object) # training the model
    
    # visualization of training and loss over all epochs
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(LOSS_PER_EPOCH, label='Training Loss')
    plt.legend(loc='lower right')
    plt.ylabel('Categorical Cross Entropy Loss')
    #plt.ylim([min(plt.ylim()),1])
    plt.title('Training Loss')

    plt.subplot(2, 1, 2)
    plt.plot(ACCURACY_PER_EPOCH, label='Training Accuracy')
    plt.legend(loc='upper right')
    plt.ylabel('Accuracy')
    #plt.ylim([0,1.0])
    plt.title('Training Accuracy')
    plt.xlabel('epoch')
    plt.show()


if __name__ == "__main__":
    main()