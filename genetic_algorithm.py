import numpy as np
import neural_network as nn

class GeneticAlgorithm():
    def __init__(self, initial_mutation_rate :float, mutation_rate_step :float, n_pairs :int, n_population :int, elitism :bool, neural_network_shapes :list):
        self.initial_mutation_rate = initial_mutation_rate
        self.mutation_rate_step = mutation_rate_step
        self.n_pairs = n_pairs
        self.n_population = n_population
        self.elitism = elitism
        self.mutation_rate = initial_mutation_rate
        self.batch_size = n_population//n_pairs
        self.best_fitness = 0
        self.best_model = None
        self.network_shapes = neural_network_shapes
    

    def __mutate_neurons(self, weights :list, n_mutations_neurons: int):   
        # Get the index of the neurons to mutate
        mutations_idx = np.random.choice(len(weights), n_mutations_neurons, replace=False)

        # Create mutations
        for i in range(0, len(mutations_idx)):
            weights[mutations_idx[i]] = np.random.rand()

        return weights


    def __make_mutations(self, offspring_weights):
        # Get the quantity of individuals to mutate
        n_mutations = self.mutation_rate * self.n_population
        n_mutations = int(n_mutations)

        # Get the quantity of neurons to mutate
        n_mutations_neurons = self.mutation_rate * len(offspring_weights[0])
        n_mutations_neurons = int(n_mutations_neurons)

        # Get the index of the individuals to mutate
        mutations_idx = np.random.choice(self.n_population, n_mutations, replace=False)

        # Create mutations
        for i in range(0, len(mutations_idx)):
            idx = mutations_idx[i]
            offspring_weights[idx] = self.__mutate_neurons(offspring_weights[idx], n_mutations_neurons)

        return offspring_weights


    def __make_cross_over(self, networks :list, fitness_sorted_indexes :list):
        offspring = []

        # Select individuals to reproduce
        probabilities = np.arange(1,len(fitness_sorted_indexes)+1)[::-1]
        probabilities = probabilities / sum(probabilities)
        parents = np.random.choice(fitness_sorted_indexes, self.n_pairs*2, p=probabilities,replace=False)

        # Create offspring using pairs of individuals
        for i in range(0, len(parents), 2):
            neurons_parent_one = networks[parents[i]].get_neurons()
            neurons_parent_two = networks[parents[i+1]].get_neurons()

            # Generate the offspring of the selected pair
            for j in range(0, self.batch_size):
                offspring_individual = self.__cross_over(neurons_parent_one, neurons_parent_two)
                offspring.append(offspring_individual)

        return offspring


    def __cross_over(self, neurons_one :list, neurons_two :list):
        # Create copy of the first individual
        new_individual = neurons_one.copy()

        # Get the quantity of copied neurons from the second individual
        qnt_neurons = np.random.randint(low=0,high=len(neurons_one))

        # Get the indexes to replace
        idx = np.random.choice(len(neurons_one), qnt_neurons, replace=False)

        # Copies the weight values
        for i in range(0, qnt_neurons):
            new_individual[idx[i]] = neurons_two[idx[i]]
        
        return new_individual


    def __compare_best(self, fitness :list, individuals :list):
        # Gets Best Individuals id's List
        idx = np.argsort(fitness)[::-1]
        best_individual = individuals[idx[0]]

        # If has new best Individual
        if fitness[idx[0]] > self.best_fitness:
            # Saves Best Individuals
            self.best_model = best_individual
            self.mutation_rate = self.initial_mutation_rate
            self.best_fitness = fitness[idx[0]]

        # Else , Increase Mutation Rate
        else:
            self.mutation_rate = self.mutation_rate + self.mutation_rate_step

            if self.mutation_rate > 1:
                self.mutation_rate = 1

        return idx


    def __create_models(self, offspring :list):
        models = []
        for i in range(0, len(offspring)):
            new_model = nn.NeuralNetwork().create_from_weights_list(offspring[i], self.network_shapes)
            models.append(new_model)

        return models


    def breed(self, fitness:list, networks:list):
        # Gets Best Individuals id's List
        idx = self.__compare_best(fitness, networks)

        # Makes Cross Over
        offspring = self.__make_cross_over(networks, idx)

        # Makes Mutations
        offspring = self.__make_mutations(offspring)

        # Recreate Models
        offspring = self.__create_models(offspring)

        # Check for Elitism
        if self.elitism:
            if self.best_model != None:
                offspring[0] = self.best_model

        return offspring