import copy
import random
import math

# Support for type hints
from typing import List, Sequence, Callable

class ObjectiveFunction:
    # Get a SystemRandom instance out of random package
    system_random = random.SystemRandom();

    def __init__(self,
                 aNumberOfDimensions: int,
                 aBoundarySet: List[List[float]],
                 anObjectiveFunction: Callable[[List[float]], float],
                 aFlag: int=0):

        # aFlag is 1 for Minimisation
        # aFlag is 2 for Maximisation
        self.boundary_set = copy.deepcopy(aBoundarySet);
        self.number_of_dimensions = aNumberOfDimensions;
        self.objective_function = anObjectiveFunction;
        self.number_of_evaluation = 0;
        self.flag = aFlag;
        self.global_optimum = None;
        self.verbose = False;   # Use for debugging

    def initialRandomGuess(self):
        if self.number_of_dimensions == 1:
            return ObjectiveFunction.system_random.uniform(self.boundary_set[0][0], self.boundary_set[0][1]);
        else:
            guess = [];
            for i in range(self.number_of_dimensions):
                guess.append(ObjectiveFunction.system_random.uniform(self.boundary_set[i][0], self.boundary_set[i][1]))
            return guess;

    def initialGuess(self):
        if self.number_of_dimensions == 1:
            return self.boundary_set[0][0] + (self.boundary_set[0][1] - self.boundary_set[0][0]) / 2;
        else:
            guess = [];
            for i in range(self.number_of_dimensions):
                guess.append(self.boundary_set[i][0] + (self.boundary_set[i][1] - self.boundary_set[i][0]) / 2);
            return guess;

    def minimisationFunction(self, aParameterSet: List[float]):
        return self.evaluate(aParameterSet, 1)

    def maximisationFunction(self, aParameterSet: List[float]):
        return self.evaluate(aParameterSet, 2)

    def evaluate(self, aParameterSet: List[float], aFlag: int):
        self.number_of_evaluation += 1;

        objective_value = self.objective_function(aParameterSet);
        if aFlag != self.flag:
            objective_value *= -1;

        return objective_value;

    def getDistanceToGlobalOptimum(self, aParameterSet: List[float]) -> float:

        if self.global_optimum == None:
            return float('NaN');

        distance = 0.0;
        for t, r in zip(aParameterSet, self.global_optimum):
            distance += math.pow(t - r, 2);

        return math.sqrt(distance);
