import numpy as np

class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_dense_layer(self, in_size, out_size):
        layer = np.random.rand(in_size,out_size)
        self.layers.append(layer)

    def predict(self, x :list):
        for j in range(0, len(self.layers)):
            x = np.dot(x, self.layers[j])
        return x

    def get_shapes(self):
        shapes = []
        for l in self.layers:
            shapes.append(l.shape)
        return shapes

    def get_neurons(self):
        neurons = []
        for l in self.layers:
            c = np.ndarray.flatten(l)
            neurons.append(c)

        neurons = np.concatenate(neurons, axis=0)
        return neurons

    def create_from_shapes(self, shapes :list):
        for s in shapes:
            self.add_dense_layer(s[0],s[1])
        return self

    def create_from_weights_list(self, weights :list,shapes :list):
        cur_idx = 0
        new_layers = []
        for shape in shapes:
            new_idx = cur_idx + (shape[0]*shape[1])
            aux = weights[cur_idx:new_idx]
            aux = np.reshape(aux, shape)
            new_layers.append(aux)
            cur_idx = new_idx

        self.layers = new_layers
        return self