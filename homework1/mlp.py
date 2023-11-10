from layer import Layer


class MLP:

    def __init__(self, num_of_layers: int, size_of_layers: []):
        self.di_list = []  # List of activations
        self.layers = []
        self.num_layers = len(size_of_layers)
        if num_of_layers == 1:
            self.layers.append(Layer("softmax", 10, 64))
        else:
            for i in range(0, num_of_layers):

                if i == 0:
                    self.layers.append(Layer("sigmoid", size_of_layers[i], 64))
                elif i == num_of_layers - 1:
                    self.layers.append(Layer("softmax", 10, size_of_layers[i - 1]))
                else:
                    self.layers.append(Layer('sigmoid', size_of_layers[i], size_of_layers[i - 1]))

    def forward(self, input_matrix):
        self.di_list.append(input_matrix)
        inputs = input_matrix
        for index, layer in enumerate(self.layers):
            inputs = layer.forward(inputs, self.di_list)
        return inputs


    def backward(self, dcce):
        input = dcce
        for index, layer in enumerate(reversed(self.layers)):
            input = layer.backward(self.di_list, input, self.num_layers - index)
        self.di_list = []