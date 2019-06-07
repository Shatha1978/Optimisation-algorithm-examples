import math

from ObjectiveFunction import *

def objectiveFunction(aSolution):

    # Function:
    #   f(x,y)=-20&amp;\exp \left[-0.2{\sqrt {0.5\left(x^{2}+y^{2}\right)}}\right]\\&amp;{}-\exp \left[0.5\left(\cos 2\pi x+\cos 2\pi y\right)\right]+e+20}}
    # In LaTeX

    cost = 0.0;
    cost += -20.0 * math.exp(0.2 * -(math.sqrt(math.pow(aSolution[0], 2) + math.pow(aSolution[1], 2))));
    cost += - math.exp(0.5 * (math.cos(2.0 * math.pi * aSolution[0]) + math.cos(2.0 * math.pi * aSolution[1])));
    cost += math.e + 20.0;

    return cost;

class AckleyFunction(ObjectiveFunction):
    def __init__(self):

        number_of_dimensions = 2;

        self.boundaries = [];
        for i in range(number_of_dimensions):
            self.boundaries.append([-5,5]);

        super().__init__(number_of_dimensions,
                         self.boundaries,
                         objectiveFunction,
                         1);

        self.global_optimum = [0, 0];
