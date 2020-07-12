import numpy as np
import neural_network as nn
import genetic_algorithm as ga
from sklearn.metrics import accuracy_score as ac
import pandas as pd

np.random.seed(59)

def make_predictions(models:list, x:list):
    preds = []
    for i in range(0, len(models)):
        y = models[i].predict(x)
        preds.append(y)
    return preds

def make_model():
    model = nn.NeuralNetwork()
    model.add_layer(15,8, nn.Activations().sigmoid)
    model.add_layer(8,4, nn.Activations().sigmoid)
    model.add_layer(4,2, nn.Activations().sigmoid)
    model.add_layer(2,1, nn.Activations().softmax)
    return model

def make_models(n_population):
    models = []
    for i in range(0, n_population):
        model = make_model()
        models.append(model)
    return models

def get_fitness(y_true, preds):
    fitness = []
    for i in range(0, len(preds)):
        acc = ac(y_true, preds[i])
        #print(acc)
        fitness.append(acc)
    return fitness


x = pd.read_csv('train_mod.csv')
y = np.array(x['Survived'])
x = x.drop('Survived', axis=1)
x = np.array(x)


#x = np.load('X.npy')
#y = np.load('Y.npy')
# Baseline = 0.5351

print(x.shape)
print(y.shape)

n_population = 10000
GA = ga.GeneticAlgorithm(0.1, 0.1, 1000, n_population, True)
models = make_models(n_population)


pred = np.zeros(len(x))
baseline = get_fitness(y, [pred])
print(baseline[0])

count = 1
while 1:
    preds = make_predictions(models, x)
    fitness = get_fitness(y, preds)
    GA.breed(fitness, models)
    print('Epoch:%i Accuracy:%.3f' % (count, fitness[0]))
    count += 1