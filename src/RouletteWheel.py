import random

from SelectionOperator import *

class RouletteWheel(SelectionOperator):

    def __init__(self):
        super().__init__("Roulette wheel selection");
        self.sum_fitness = 0.0

        # Get a SystemRandom instance out of random package
        self.system_random = random.SystemRandom();

    def preProcess(self, anIndividualSet):
        # Compute fitness sumation
        self.sum_fitness = 0.0
        for individual in anIndividualSet:
            self.sum_fitness += individual.fitness

    def __select__(self, anIndividualSet, aFlag): # aFlag == True for selecting good individuals,
                                                  # aFlag == False for selecting bad individuals,


        if aFlag == False:
            raise NotImplementedError("Selecting a bad individual is not implemented in RouletteWheel!")

        # Random number between(0 - self.sum_fitness)
        random_number = self.system_random.uniform(0.0, self.sum_fitness)

        # Select the individual depending on the probability
        accumulator = 0.0;
        for individual in anIndividualSet:
            accumulator += individual.fitness;
            if accumulator >= random_number:
                return anIndividualSet.index(individual)
