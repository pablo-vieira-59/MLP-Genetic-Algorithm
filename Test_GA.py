import numpy as np
import neural_network as nn
import genetic_algorithm as ga

np.random.seed(59)

def make_predictions(models:list, x:list):
    for i in range(0, len(models)):
        y = models[i].predict(x)
    
        n = models[i].get_neurons()
        for neuron in n:
            print(neuron.weights)
        print('\n')

def make_model():
    model = nn.NeuralNetwork()
    model.add_layer(4)
    model.add_layer(4)
    model.add_layer(1)
    return model

def change_weights(model, value):
    for layer in model.layers:
        for neuron in layer.neurons:
            for i in range(0, len(neuron.weights)):
                neuron.weights[i] = value
    return model

def make_models(n_population):
    models = []
    for i in range(0, n_population):
        model = make_model()
        model = change_weights(model, i+1)
        models.append(model)
    return models


n_population = 10
GA = ga.GeneticAlgorithm(0.5, 0.1, 2, n_population, False)
models = make_models(n_population)
x = [1,2,3,4]

make_predictions(models, x) 
fitness = np.random.randint(10,size=10)
GA.breed(fitness, models)

make_predictions(models, x)
fitness = np.random.randint(10,size=10)
GA.breed(fitness, models)

make_predictions(models, x)
fitness = np.random.randint(10,size=10)
GA.breed(fitness, models)

