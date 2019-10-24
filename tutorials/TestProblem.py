import math

from ObjectiveFunction import *


class TestProblem(ObjectiveFunction):
    def __init__(self):

        number_of_dimensions = 2;

        self.boundaries = [];
        for _ in range(number_of_dimensions):
            self.boundaries.append([-32.768, 32.768]);

        super().__init__(number_of_dimensions,
                         self.boundaries,
                         self.objectiveFunction,
                         ObjectiveFunction.MINIMISATION);

        self.name = "Ackley Function";

        self.a = 20;
        self.b = 0.2;
        self.c = 2 * math.pi;


    def objectiveFunction(self, aSolution):

        M = 0;
        N = 0;
        O = 1 / self.number_of_dimensions;

        for i in range(self.number_of_dimensions):
            M += math.pow(aSolution[i], 2);
            N += math.cos(self.c * aSolution[i]);

        return -self.a * math.exp(-self.b * math.sqrt(O * M)) - math.exp(O * N) + self.a + math.e;
