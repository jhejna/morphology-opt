import os
import random
from optimal_agents.utils.loader import get_morphology
import numpy as np

class Individual(object):
    '''
    Defines an individual in the population.
    Extra info can be added in the `info` attribute.
    Made a class instead of named tuple for more flexibility 
    later if needed.
    '''
    def __init__(self, morphology, index=None, model=None, fitness=-float('inf'), parent=None, age=0):
        self.morphology = morphology
        self.index = index
        self.model = model
        self.parent = parent
        self.fitness_list = []
        self.fitness = fitness
        self.age = age
        self.info = dict()
    
    def update(self, fitness, model):
        ''' Updates the individual after training '''
        self.fitness_list.append(fitness)
        self.fitness = fitness
        self.model = model
        self.age += 1 # Increment the age after training.

class EvoAlg(object):

    def __init__(self, params, keep_percent=0.2, random_percent=0.2, new_percent=0.05, 
                        pruning_multiplier=1, pruning_start=1, retrain=False, fitness_window=1, eval_envs=None):
        
        self.params = params # Don't like that I save the params object, but what are you gonna do?
        self.morphology_kwargs = params['morphology_kwargs']
        self.mutation_kwargs = params['mutation_kwargs']
        self.keep_percent = keep_percent
        self.random_percent = random_percent
        self.new_percent = new_percent
        self.retrain = retrain
        self.pruning_multiplier = pruning_multiplier
        self.pruning_model = None
        self.pruning_start = pruning_start
        self.fitness_window = fitness_window
        self.eval_envs = eval_envs
    
    def _train_policies(self, gen_idx):
        ''' Updates self.population with correct fitness and model values'''
        ''' must call individual.update on all morphologies '''
        raise NotImplementedError
    
    def _mutate(self, individual):
        '''
        mutates an individual morphology
        returns a new Individual object.
        '''
        raise NotImplementedError
    
    def _pruning_update(self, gen_idx):
        '''
        Update the pruning model
        Method not required.
        '''
        pass

    def _clean(self, gen_idx):
        '''
        called at the end of a generation once the population has been updated.
        Can be used for deleting files etc.
        '''
    
    def mutate(self, individual):
        # Wrapper function around self.mutate to propagate information to the new morphology correctly.
        new_individual = self._mutate(individual)
        if not new_individual is None:
            new_individual.parent = individual.index
            new_individual.fitness_list = individual.fitness_list.copy()
            if not self.retrain:
                new_individual.model = individual.model # Assume they inherit the same model.
        return new_individual

    def sample_index(self, length):
        # Can be overriden to produce other types of sampling behavior. 
        # By default just return uniform.
        # Population assumed to be sorted in reverse fitness (highest first)
        return random.randint(0, length-1)

    def compute_fitness(self, individual):
        return np.mean(individual.fitness_list[-self.fitness_window:])

    def sort_population(self):
        self.population.sort(key=lambda individual: self.compute_fitness(individual), reverse=True)

    def select(self, population, num, gen_idx):
        if num == 0:
            return []
        mutated_individuals = []
        num_to_generate = num
        if not self.pruning_model is None and gen_idx > self.pruning_start:
            num_to_generate = int(num_to_generate * self.pruning_multiplier)
        for _ in range(num_to_generate):
            individual = population[self.sample_index(len(population))]
            if individual is None:
                    mutated_individuals.append(Individual(get_morphology(self.params)))
            else:
                mutated_individuals.append(self.mutate(individual))

        if self.pruning_model is None:
            return mutated_individuals[:num]
        else:
            value_scores = [None for _ in range(len(mutated_individuals))]
            for i, individual in enumerate(mutated_individuals):
                value_scores[i] = self.pruning_model.evaluate(individual.morphology) # Passes in the morphology object.
            scored_pairs = list(zip(mutated_individuals, value_scores))
            scored_pairs.sort(key=lambda x: x[1], reverse=True)
            mutated_individuals, _ = zip(*scored_pairs)
            mutated_individuals = list(mutated_individuals)
            return mutated_individuals[:num]
    
    def learn(self, path, population_size, num_generations):
        num_keep = int(self.keep_percent * population_size)
        num_random = int(self.random_percent * population_size)
        num_new = int(self.new_percent * population_size)
        num_top_mutated = population_size - num_keep - num_random - num_new

        # init the population. As everything is None, must add indices here.
        self.population = [Individual(get_morphology(self.params), index=i) for i in range(population_size)]
        # Init path in case train policies needs it
        self.path = path
        os.makedirs(self.path, exist_ok=True)

        for gen_idx in range(num_generations):
            self._train_policies(gen_idx)
            # Sort the population by measured fitness
            self.sort_population()
            # Store data about the generation in a txt file:
            with open(os.path.join(path, "gen" + str(gen_idx) + ".txt"), "w+") as f:
                for individual in self.population:
                    f.write(str(self.compute_fitness(individual)) + " " + str(individual.index) + "\n")
            
            #  Remove invalid individuals from the population if they failed training.
            for i, individual in enumerate(self.population):
                if individual.fitness == -float('inf'): # We don't need to resort as they will be at the end anyways.
                    self.population[i] = Individual(get_morphology(self.params), index=i) # Construct a new individual.

            # Update the population
            self._pruning_update(gen_idx)

            new_population = [] 

            # Carry over the kept species
            new_population.extend(self.population[:num_keep])
            # Mutate from the top individuals
            new_population.extend(self.select(self.population[:num_keep], num_top_mutated, gen_idx)) # Mutate more from the top individuals.
            # Randomly mutate from other individuals in the population
            new_population.extend(self.select(self.population[num_keep:], num_random, gen_idx))
            # Add new morphologies
            new_population.extend(self.select([None], num_new, gen_idx)) # List only contains "None" morphology.

            assert len(new_population) == len(self.population), "Populations were different lengths"
            
            # Index all of the new morphologies
            for i, individual in enumerate(new_population):
                individual.index = i

            self.population = new_population

            self._clean(gen_idx)

        best_morphology = max(self.population, key=lambda x: x.fitness).morphology
        best_morphology.save(os.path.join(self.path, "best_morphology.pkl"))
