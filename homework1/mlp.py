from layer import Layer


class MLP:

    def __init__(self, num_of_layers: int, size_of_layers: []):
        """
        Initializes a Multi-Layer Perceptron (MLP) with specified layer sizes and activation functions.

        :param num_of_layers: The number of layers in the MLP.
        :param size_of_layers: A list containing the size of each layer.
        """

        self.di_list = []  # List of activations during forward pass
        self.layers = []  # List to store the layers of the MLP
        self.num_layers = len(size_of_layers)

        # Check if the MLP has only one layer (output layer)
        if num_of_layers == 1:
            self.layers.append(Layer("softmax", 10, 64))
        else:
            # Iterate over the specified number of layers
            for i in range(0, num_of_layers):
                # Check if it is the first hidden layer
                if i == 0:
                    self.layers.append(Layer("sigmoid", size_of_layers[i], 64))
                # Check if it is the last layer (output layer)
                elif i == num_of_layers - 1:
                    self.layers.append(Layer("softmax", 10, size_of_layers[i - 1]))
                # For all other hidden layers
                else:
                    self.layers.append(Layer('sigmoid', size_of_layers[i], size_of_layers[i - 1]))

    def forward(self, input_matrix):
        """
        Performs the forward pass through the MLP.

        :param input_matrix: The input matrix to the MLP.
        :return: The output of the MLP after the forward pass.
        """

        self.di_list.append(input_matrix)  # Store network input as activation for first layer
        inputs = input_matrix

        # Iterate over the layers and perform forward pass
        for index, layer in enumerate(self.layers):
            inputs = layer.forward(inputs, self.di_list)

        return inputs


    def backward(self, dcce):
        """
        Performs the backward pass through the MLP to update parameters.

        :param dcce: The gradient of the categorical cross-entropy loss.
        """

        input = dcce

        # Iterate over the layers in reverse order and perform backward pass
        for index, layer in enumerate(reversed(self.layers)):
            input = layer.backward(self.di_list, input, self.num_layers - index)

        self.di_list = []  # Clear the list of activations for the next iteration