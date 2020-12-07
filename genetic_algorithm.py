import numpy as np
import neural_network as nn

class GeneticAlgorithm():
    _best_model = None
    _best_fitness = 0
    _mutation_rate = 0.1
    _batch_size = 0

    initial_mutation_rate = 0.1
    mutation_rate_step = 0.1
    n_pairs = 0
    n_population = 0
    elitism = True

    def __init__(self, initial_mutation_rate :float, mutation_rate_step :float, n_pairs :int, n_population :int, elitism :bool):
        self.initial_mutation_rate = initial_mutation_rate
        self.mutation_rate_step = mutation_rate_step
        self.n_pairs = n_pairs
        self.n_population = n_population
        self.elitism = elitism
        self._mutation_rate = initial_mutation_rate
        self._batch_size = n_population//n_pairs
        self._best_fitness = 0
        self._best_model = None
    

    def mutate_neurons(self, offspring_individual :list, n_mutations_neurons: int):   
        mutations_idx = np.random.choice(len(offspring_individual), n_mutations_neurons, replace=False)
        for i in range(0, len(mutations_idx)):
            neuron = offspring_individual[mutations_idx[i]]
            neuron = (np.random.randint(low=-10000,high=10000,size=len(neuron)) * 0.0001).tolist()
            offspring_individual[mutations_idx[i]] = neuron

        return offspring_individual


    def make_mutations(self, offspring):
        n_mutations = self._mutation_rate * self.n_population
        n_mutations = int(n_mutations)

        n_mutations_neurons = self._mutation_rate * len(offspring[0])
        n_mutations_neurons = int(n_mutations_neurons)

        mutations_idx = np.random.choice(self.n_population, n_mutations, replace=False)
        for i in range(0, len(mutations_idx)):
            offspring[mutations_idx[i]] = self.mutate_neurons(offspring[mutations_idx[i]], n_mutations_neurons)

        return offspring


    def make_cross_over(self, individuals :list, idx :list):
        offspring = []
        for i in range(0, self.n_pairs * 2, 2):
            neurons_parent_one = individuals[idx[i]].get_neurons()
            neurons_parent_two = individuals[idx[i+1]].get_neurons()
            for j in range(0, self._batch_size):
                offspring_individual = self.cross_over(neurons_parent_one, neurons_parent_two)
                offspring.append(offspring_individual)
        return offspring


    def cross_over(self, neurons_one :list, neurons_two :list):
        # Create copy of the first individual
        new_individual = neurons_one.copy()

        # Random copy neurons from second individual
        qnt_neurons = np.random.randint(low=1,high=len(neurons_one))
        idx = np.random.choice(len(neurons_one), qnt_neurons, replace=False)
        for i in range(0, qnt_neurons):
            new_individual[idx[i]] = neurons_two[idx[i]].copy() 
        
        return new_individual


    def compare_best(self, fitness :list, individuals :list):
        # Gets Best Individuals id's List
        idx = np.argsort(fitness)[::-1]
        best_individual = individuals[idx[0]]
        # If has new best Individual
        if fitness[idx[0]] > self._best_fitness:
            # Saves Best Individuals
            self._best_model = best_individual
            self._mutation_rate = self.initial_mutation_rate
            self._best_fitness = fitness[idx[0]]
        # Else , Increase Mutation Rate
        else:
            new_mutation_rate = self._mutation_rate + self.mutation_rate_step
            if new_mutation_rate <= 0.9:
                self._mutation_rate = self._mutation_rate + self.mutation_rate_step
            else:
                self._mutation_rate = 1
        return idx


    def create_model(self, offspring_individual :list, individual :nn.NeuralNetwork):
        new_model = nn.NeuralNetwork()
        new_model.activations = individual.activations
        for i in range(0, len(offspring_individual)):
            offspring_individual[i] = np.array(offspring_individual[i])

        for layer in individual.layers:
            qnt_neurons = len(layer)
            layer_neurons = offspring_individual[0:qnt_neurons]
            new_layer = np.reshape(layer_neurons, layer.shape)
            new_model.layers.append(new_layer)

            offspring_individual = offspring_individual[qnt_neurons:len(offspring_individual)]
        return new_model


    def make_models(self, offspring :list, individuals :list):
        for i in range(0, len(individuals)):
            individuals[i] = self.create_model(offspring[i], individuals[i])
        return individuals


    def breed(self,fitness:list,individuals:list):
        # Gets Best Individuals id's List
        idx = self.compare_best(fitness, individuals)

        # Makes Cross Over
        offspring = self.make_cross_over(individuals , idx)

        # Makes Mutations
        offspring = self.make_mutations(offspring)

        # Recreate Models
        offspring = self.make_models(offspring, individuals)

        # Check for Elitism
        if self.elitism:
            if self._best_model != None:
                offspring[0] = self._best_model

        return offspring
