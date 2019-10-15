# Import the random package to generate random solutions within boundaries
import random

# Import the copy package to deep copies
#import copy

# Import the superclass (also called base class), which is an abstract class,
# to implement the subclass Individual
import Solution

# The subclass that inherits of Solution
class Individual(Solution.Solution):

    '''
    Class to handle solutions when an Evolutionary algorithm is used.
    This subclass inherits of Solution.
    '''

    # Get a SystemRandom instance out of random package
    system_random = random.SystemRandom();

    def __init__(self, anObjectiveFunction, aParameterSet = None, aComputeFitnessFlag = False):
        '''
        Constructor

        Parameters:
        anObjectiveFunction (function): the callback corresponding to the objective function
        aParameterSet (array of float): the solutino parameters (default: None)
        aComputeObjectiveFlag (bool): compute the objective value in the constructor when the Solution is created (default: False)
        '''

        super().__init__(anObjectiveFunction, 2, aParameterSet, aComputeFitnessFlag); # 2 for maximisation

    def copy(self):
        '''
        Create a copy of the current solution

        Returns:
        Solution: the new copy
        '''

        temp = Individual(self.objective_function, self.parameter_set, False);
        temp.objective = self.objective;
        return temp;
