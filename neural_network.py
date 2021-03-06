import numpy as np

class Activations():
    def one_to_one(self, x :list):
        x = np.multiply(x, -1)
        x = np.exp(x)
        x = 1 / (1 + x)
        x = (x - 0.5) / 0.5
        return x

    def sigmoid(self, x :list):
        x = np.multiply(x, -1)
        x = np.exp(x)
        x = 1 / (1 + x)
        return x

    def linear(self, x :list):
        return x

    def binary(self, x :list):
        x = np.where(x > 0, 1, 0)
        return x

class NeuralNetwork():
    layers = []
    activations = []
    
    def __init__(self):
        self.layers = []
        self.activations = []

    def add_layer(self, in_size, out_size, activation):
        layer = np.random.randint(low=-1000,high=1000,size=(in_size,out_size)) * 0.0001
        self.layers.append(layer)
        self.activations.append(activation)

    def predict(self, x :list):
        for j in range(0, len(self.layers)):
            x = np.dot(x, self.layers[j])
            if j < len(self.layers) - 1:
                x = x + 1
            x = self.activations[j](x)
        return x

    def get_neurons(self):
        neurons = []
        for layer in self.layers:
            for neuron in layer:
                neurons.append(neuron)
        return neurons