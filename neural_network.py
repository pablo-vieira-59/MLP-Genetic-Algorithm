import numpy as np

class Neuron():
    weights = []

    def __init__(self):
        self.create_weights(n_connections=0)

    def create_weights(self, n_connections :int):
        self.weights = []
        if n_connections == 0:
            self.weights.append(1)
        else:
            arr = np.random.random(n_connections)
            arr = np.round(arr, 5)
            self.weights = (2 * arr) - 1

    def calculate(self, x):
        results = []
        if len(self.weights) == 0:
            results.append(x)
        else:
            results = np.multiply(self.weights, x)
        return results

class Layer():
    neurons = []
    bias = 0

    def __init__(self, n_neurons):
        self.neurons = []
        self.bias = 0
        for i in range(0, n_neurons):
            neuron = Neuron()
            self.neurons.append(neuron)
    
    def forward(self, x :list):
        Y = np.zeros(len(self.neurons[0].weights))
        for i in range(0,len(self.neurons)):
            y = self.neurons[i].calculate(x[i])
            for j in range(0, len(y)):
                Y[j] = y[j] + self.bias
        
        return Y

class Activations():
    def sigmoid(self, x:list):
        x = np.multiply(x, -1)
        x = np.exp(x)
        x = 1 / (1 + x)
        return x

    def one_to_one(self, x:list):
        x = np.multiply(x, -1)
        x = np.exp(x)
        x = 1 / (1 + x)
        x = (x - 0.5) / 0.5
        return x

class NeuralNetwork():
    layers = [] 
    activations = Activations()
    
    def __init__(self):
        self.layers = []
        self.activations = Activations()

    def add_layer(self, n_neurons):
        layer = Layer(n_neurons)
        if len(self.layers) > 0:
            previous_layer = self.layers[len(self.layers)-1]
            #bias = np.round(np.random.random(), 5)
            #previous_layer.bias = 2 * bias - 1
            bias = 0
            for i in range(0, len(previous_layer.neurons)):
                previous_layer.neurons[i].create_weights(n_neurons)
        self.layers.append(layer)

    def predict(self, x:list):
        for i in range(0, len(self.layers)-1):
            x = self.layers[i].forward(x)   
            x = np.round(x, 3)
        x = self.activations.one_to_one(x)
        return x

    def predict_many(self, x:list):
        predictions = []
        for i in range(len(x)):
            y = self.predict(x[i])
            predictions.append(y)
        return predictions

    def get_neurons(self):
        neurons = []
        for i in range(0, len(self.layers)):
            neurons = np.concatenate([neurons, self.layers[i].neurons])
        return neurons
