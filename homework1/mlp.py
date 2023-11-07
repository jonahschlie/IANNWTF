from layer import Layer


class MLP:

    def __init__(self, num_of_layers: int, size_of_layers: []):

        self.layers = []
        for i in range(0, num_of_layers):
            if i == 0:
                self.layers.append(Layer("sigmoid", size_of_layers[i], 64))
            elif i == num_of_layers - 1:
                self.layers.append(Layer("softmax", 10, size_of_layers[i - 1]))
            else:
                self.layers.append(Layer('sigmoid', size_of_layers[i], size_of_layers[i - 1]))

    def forward(self, input_matrix):
        inputs = input_matrix
        output = []
        for layer in self.layers:
            output = layer.forward(inputs)
            inputs = output
        return output
