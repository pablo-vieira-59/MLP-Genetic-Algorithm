import numpy as np
import random as rd

class GeneticAlgorithm():

    decimal = 3
    input_size = 0
    hidden_size = 0
    hidden_neurons = 0
    out_size = 0
    population_size = 0
    mutation_per = 0
    _best_model = None
    _best_fitness = 0

    def __init__(self, input_size :int, hidden_size :int, hidden_neurons :int, out_size :int, population_size :int, mutation_per :int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_neurons = hidden_neurons
        self.out_size = out_size
        self.population_size = population_size
        self.mutation_per = mutation_per
         
     
    def init_layer(self):
        input_layer = []
        for i in range(0,self.input_size):
            layer_i = []
            for j in range(0,self.hidden_neurons):
                layer_i.append(0)
            input_layer.append(layer_i)
            
        hidden_layers = []
        for i in range(0,self.hidden_size + 2):
            layer_i = []
            for j in range(0,self.hidden_neurons):
                layer_i.append(0)
            hidden_layers.append(layer_i)
        
        output_layer = []
        for i in range(0,self.hidden_neurons):
            layer_i = []
            for j in range(0,self.out_size):
                layer_i.append(0)
            output_layer.append(layer_i)
            
        all_weights = [input_layer,hidden_layers,output_layer]
        return all_weights


    def init_bias(self):
        biases = []
        for i in range(0,self.hidden_size):
            bias = []
            for j in range(0,self.hidden_neurons):
                bias.append(0)
            biases.append(bias)
        out_bias = []
        for i in range(0,self.out_size):
            out_bias.append(0)
        biases.append(out_bias)
        
        return biases


    def select_mates(self,fitness:list):
        # Selecting Individual 1
        individual_1_1 = 0
        individual_1_2 = 0
        while individual_1_1 == individual_1_2:
            individual_1_1 = rd.randint(0,len(fitness)-1)
            individual_1_2 = rd.randint(0,len(fitness)-1)
            
        individual_1 = self.competition(individual_1_1,individual_1_2,fitness)
        
        # Selecting Individual 2
        individual_1_1 = 0
        individual_1_2 = 0
        while individual_1_1 == individual_1_2 or individual_1_1 == individual_1 or individual_1_2 == individual_1:
            individual_1_1 = rd.randint(0,len(fitness)-1)
            individual_1_2 = rd.randint(0,len(fitness)-1)
            
        individual_2 = self.competition(individual_1_1,individual_1_2,fitness)
        
        pair = [individual_1,individual_2]
        return pair
            
            
    def competition(self,individual_1,individual_2,fitness:list):
        winner = 0
        if fitness[individual_1] > fitness[individual_2]:
            winner = individual_1
        else:
            winner = individual_2
        return winner


    def get_best(self,fitness:list):
        copy = fitness.copy()
        copy.sort(reverse=True)
        weights = []
        for i in range(len(copy)):
            weights.append(fitness.index(copy[i]))
        return weights


    def gen_mutation(self,layers):
        pos_1 = rd.randint(0, len(layers)-1)
        pos_2 = rd.randint(0, len(layers[pos_1])-1)
        values = []
        for i in range(0,len(layers[pos_1][pos_2])):
            value = np.round(rd.uniform(-1, 1), self.decimal)
            values.append(value)
        layers[pos_1][pos_2] = values
        return layers


    def gen_mutation_weak(self,layers):
        pos_1 = rd.randint(0, len(layers)-1)
        pos_2 = rd.randint(0, len(layers[pos_1])-1)
        pos_3 = rd.randint(0, len(layers[pos_1][pos_2])-1)
        layers[pos_1][pos_2][pos_3]
        return layers


    def gen_mutation_bias(self,biases):
        pos_1 = rd.randint(0, len(biases)-1)
        for i in range(0,len(biases[pos_1])):
            value = np.round(rd.uniform(-1, 1), self.decimal)
            biases[pos_1][i] = value
        return biases


    def gen_mutation_bias_weak(self,biases):
        pos_1 = rd.randint(0, len(biases)-1)
        pos_2 = rd.randint(0, len(biases[pos_1])-1)
        value = np.round(rd.uniform(-1, 1), self.decimal)
        biases[pos_1][pos_2] = value
        return biases


    def gen_weights(self):
        weights = self.init_layer()
        for k in range(0, len(weights)):
            for j in range(0, len(weights[k])):
                for i in range(0, len(weights[k][j])):
                    weights[k][j][i] = np.round(rd.uniform(-1, 1), self.decimal)

        return weights


    def gen_biases(self):
        biases = self.init_bias()
        for i in range(0,len(biases)):
            for j in range(0,len(biases[i])):
                biases[i][j] = np.round(rd.uniform(-1,1),self.decimal)
        return biases


    def gen_breed(self, fitness :list, individuals :list, qnt_pairs :int):
        # Getting Weights
        weights = []
        for i in range(0,len(individuals)):
            weights.append(individuals[i].coefs_)
        
        # Getting Biases
        biases = []
        for i in range(0,len(individuals)):
            biases.append(individuals[i].intercepts_)
            
        # Breeding Offspring
        offspring = []
        offspring_biases = []
        pair_offspring = self.population_size/qnt_pairs
        for i in range(0,qnt_pairs):
            individuals_id = self.select_mates(fitness)
            id_1 = individuals_id[0]
            id_2 = individuals_id[1]
            for j in range(0,int(pair_offspring)):
                offspring.append(self.cross_over(weights[id_1],weights[id_2]))
                offspring_biases.append(self.cross_over_bias(biases[id_1],biases[id_2]))
        
        # Mutating Offspring
        mutation_offspring = self.mutation_per * self.population_size
        for i in range(0,int(mutation_offspring)):
            index = rd.randint(0,len(offspring)-1)
            offspring[index] = self.gen_mutation(offspring[index])
            offspring_biases[index] = self.gen_mutation_bias(offspring_biases[index])
        
        # Keeping the best individual
        best_list = self.get_best(fitness)
        best_id = best_list[0]
        best_fit = fitness[best_id]
        if(best_fit > self._best_fitness):
            self._best_model = individuals[best_id]
            self._best_fitness = best_fit
        
        generation = [offspring,offspring_biases]
        return generation
    
    
    def gen_breed_best(self,fitness:list,individuals:list,qnt_pairs):
        # Getting Weights
        weights = []
        for i in range(0,len(individuals)):
            weights.append(individuals[i].coefs_)
        
        # Getting Biases
        biases = []
        for i in range(0,len(individuals)):
            biases.append(individuals[i].intercepts_)
            
        # Breeding Offspring
        offspring = []
        offspring_biases = []
        pair_offspring = self.population_size/qnt_pairs
        pair_id = 0
        for i in range(0,qnt_pairs):
            id_1 = self.get_best(fitness)[pair_id]
            id_2 = self.get_best(fitness)[pair_id+1]
            for j in range(0,int(pair_offspring)):
                offspring.append(self.cross_over(weights[id_1],weights[id_2]))
                offspring_biases.append(self.cross_over_bias(biases[id_1],biases[id_2]))
            pair_id += 2
        
        # Mutating Offspring
        mutation_offspring = self.mutation_per * self.population_size
        for i in range(0,int(mutation_offspring)):
            offspring[i] = self.gen_mutation(offspring[i])
            offspring_biases[i] = self.gen_mutation_bias(offspring_biases[i])
        
        # Keeping the best individual
        best_list = self.get_best(fitness)
        best_id = best_list[0]
        best_fit = fitness[best_id]
        if(best_fit > self._best_fitness):
            self._best_model = individuals[best_id]
            self._best_fitness = best_fit
        
        generation = [offspring,offspring_biases]
        return generation


    def cross_over(self,weights_one,weights_two):
        offspring = self.copy_layers(weights_one)
        max_features = self.count_features()
        qnt_features = rd.randint(1,max_features)
        for i in range(0,qnt_features):
            layer_id = rd.randint(0,len(weights_two)-1)
            neuron_id = rd.randint(0,len(weights_two[layer_id])-1)
            weight_id = rd.randint(0,len(weights_two[layer_id][neuron_id])-1)
            offspring[layer_id][neuron_id][weight_id] = weights_two[layer_id][neuron_id][weight_id]
        return offspring
    
    
    def cross_over_bias(self,bias_one,bias_two):
        offspring = self.copy_bias(bias_one)
        max_features = self.count_features_bias()
        qnt_features = rd.randint(1,max_features)
        for i in range(0,qnt_features):
            layer_id = rd.randint(0,len(bias_two)-1)
            neuron_id = rd.randint(0,len(bias_two[layer_id])-1)
            offspring[layer_id][neuron_id] = bias_two[layer_id][neuron_id]
        return offspring

        
    def count_features(self):
        features = 0
        features = (self.input_size * self.hidden_neurons) + ((self.hidden_size + 2) * self.hidden_neurons) + (self.out_size * self.hidden_neurons)
        return features
    
    
    def count_features_bias(self):
        features = 0
        features = (self.hidden_size * self.hidden_neurons) + self.out_size
        return features


    def copy_layers(self,layers):
        copy = []
        for i in range(0, len(layers)):
            layer_out = []
            for j in range(0, len(layers[i])):
                layer_mid = []
                for k in range(0, len(layers[i][j])):
                    value = 0
                    value = layers[i][j][k]
                    layer_mid.append(value)
                layer_out.append(layer_mid)
            copy.append(layer_out)

        return copy


    def copy_bias(self,bias):
        copy = []
        for i in range(0,len(bias)):
            layer = []
            for j in range(0,len(bias[i])):
                value = 0
                value = bias[i][j]
                layer.append(value)
            copy.append(layer)
                
        return copy
