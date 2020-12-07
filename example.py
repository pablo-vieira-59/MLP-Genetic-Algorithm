import numpy as np
import neural_network as nn
import genetic_algorithm as ga
from sklearn.metrics import accuracy_score as ac
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import pandas as pd
import pickle

#np.random.seed(59)

def custom_activation(x :list):
    x = np.argmax(x)
    return x

def make_predictions(models:list, x:list):
    preds = []
    for i in range(0, len(models)):
        y = models[i].predict(x)
        preds.append(y)
    return preds

def make_model():
    model = nn.NeuralNetwork()
    model.add_layer(20,20, nn.Activations().linear)
    model.add_layer(20,32, nn.Activations().linear)
    model.add_layer(32,16, nn.Activations().linear)
    model.add_layer(16,8, nn.Activations().linear)
    model.add_layer(8,3, nn.Activations().linear)
    return model

def make_models(n_population):
    models = []
    for i in range(0, n_population):
        model = make_model()
        models.append(model)
    return models

def get_fitness(y_true, preds):
    fitness = []
    winrate = []
    trades = []
    for i in range(0, len(preds)):
        wins = 0
        losses = 0
        stops = 0
        wr = 0
        for j in range(0, len(preds[i])):
            val = custom_activation(preds[i][j])
            if val == 0 or val == 1:
                if y_true[j] == val:
                    wins += 1
                else:
                    losses += 1
            elif val == 2:
                stops += 1

        score = ((wins*0.85) - losses)
        if(wins+losses) > 0:
            wr = wins / (wins + losses)

        winrate.append(wr)
        fitness.append(score * wr)
        trades.append(wins+losses)

    return fitness, winrate, trades

#x = pd.read_csv('train_mod.csv')
#y = np.array(x['Survived'])
#x = x.drop('Survived', axis=1)
#x = np.array(x)

x = np.load('../Files/X.npy')
y = np.load('../Files/Y.npy')
xv = np.load('../Files/X_v.npy')
yv = np.load('../Files/Y_v.npy')

print(x.shape)
print(xv.shape)

n_population = 100
GA = ga.GeneticAlgorithm(0.1, 0.1, 10, n_population, True)
models = make_models(n_population)

#pred = np.zeros(len(xv))
#baseline = get_fitness(yv, [pred])
#print('Baseline :',baseline[0])

count = 1
while 1:
    # Treinamento
    preds = make_predictions(models, x)
    fitness, winrates, trades = get_fitness(y, preds)
    GA.breed(fitness, models)

    #Validação
    valid = [0]
    valid_trades = [0]
    if GA._best_model != None:
        valid_pred = make_predictions([GA._best_model], xv)
        _,valid,valid_trades = get_fitness(yv, valid_pred)
    print('Epoch:%i Fitness:%.3f, Winrate:%.3f, Trades Ratio:%.3f, Validation WR:%.3f, Validation Trade Ratio:%.3f' % (count, fitness[0], winrates[0], trades[0]/len(y), valid[0], valid_trades[0]/len(yv)))
    count += 1

    # Salvando Modelo
    obj = open('../Files/GAmodel2.pkl', 'wb')
    pickle.dump(GA._best_model, obj)
    obj.close()

# Testando Modelo
obj = open('../Files/GAmodel2.pkl', 'rb')
valid_pred = make_predictions([pickle.load(obj)], xv)
valid, wr, tc = get_fitness(yv, valid_pred)
print('Fitness:%.3f, Winrate:%.3f, Trades Count:%i' % (fitness[0], wr[0], tc[0]))