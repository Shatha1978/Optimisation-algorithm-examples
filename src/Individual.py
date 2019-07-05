# Import the random package to generate random solutions within boundaries
import random

# Import the copy package to deep copies
import copy

# Import the superclass (also called base class), which is an abstract class,
# to implement the subclass Individual
import Solution

# The subclass that inherits of Solution
class Individual(Solution.Solution):

    # Get a SystemRandom instance out of random package
    system_random = random.SystemRandom();

    def __init__(self, anObjectiveFunction, aParameterSet = None):

        super().__init__(anObjectiveFunction, 2, aParameterSet); # 2 for maximisation
