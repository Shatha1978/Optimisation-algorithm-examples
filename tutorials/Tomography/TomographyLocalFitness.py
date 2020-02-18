from ObjectiveFunction import *

from TomographyGlobalFitness import TomographyGlobalFitness


class TomographyLocalFitness(ObjectiveFunction):
    def __init__(self, aGlobalFitness):

        number_of_dimensions = 2;

        self.boundaries = aGlobalFitness.boundaries;
        self.global_fitness_function = aGlobalFitness;
        self.number_of_calls = 0;

        super().__init__(number_of_dimensions,
                         self.boundaries,
                         self.objectiveFunction,
                         ObjectiveFunction.MAXIMISATION);

        self.name = "marginal fitness";


    def objectiveFunction(self, aSolution):

        self.number_of_calls += 1;

        distance_with_individual = self.global_fitness_function.global_fitness_set[-1];

        population_without_individual = [];
        individual_removed = False;

        for i,j in zip(self.global_fitness_function.current_population[0::2], self.global_fitness_function.current_population[1::2]):
            if individual_removed:
                population_without_individual.append(i);
                population_without_individual.append(j);
            else:
                if aSolution[0] == i and aSolution[1] == j:
                    individual_removed = True;
                else:
                    population_without_individual.append(i);
                    population_without_individual.append(j);

        distance_without_individual = self.global_fitness_function.objectiveFunction(population_without_individual, False);

        marginal_fitness = distance_without_individual - distance_with_individual;

        return marginal_fitness;
