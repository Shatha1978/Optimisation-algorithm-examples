"""@package ScipyMinimize
This package implements the minimize optimisers from SciPy.
@author Dr Franck P. Vidal, Bangor University
@date 5th July 2019
"""

#################################################
# import packages
###################################################
from scipy import optimize

from Solution import Solution
from Optimiser import *

## \class This class implements the simulated annealing optimisation method
class ScipyMinimize(Optimiser):

    ## \brief Constructor.
    # \param self
    # \param aCostFunction: The cost function to minimise
    def __init__(self, aCostFunction, aMethodName, tol=-1, initial_guess = None):

        super().__init__(aCostFunction, initial_guess);

        # Name of the algorithm
        self.full_name = aMethodName;
        self.short_name = aMethodName;

        self.max_iterations = -1;
        self.verbose = False;
        self.tolerance = tol;

    def setMaxIterations(self, aMaxIterations):
        self.max_iterations = aMaxIterations;

    def run(self):

        options = {'disp': self.verbose};

        if self.max_iterations > 0:
            options['maxiter'] = self.max_iterations;

        if self.tolerance > 0:
            options['ftol'] = self.tolerance;
            options['tol'] = self.tolerance;

        if self.initial_guess == None:
            self.initial_guess = self.objective_function.initialRandomGuess();

        # Methods that cannot handle constraints or bounds.
        if self.short_name == 'Nelder-Mead' or self.short_name == 'Powell' or self.short_name == 'CG' or self.short_name == 'BFGS' or self.short_name == 'COBYLA':

            result = optimize.minimize(self.objective_function.minimisationFunction,
                self.initial_guess,
                method=self.short_name,
                options=options);

        elif self.short_name == 'L-BFGS-B' or self.short_name == 'TNC' or self.short_name == 'SLSQP':
            result = optimize.minimize(self.objective_function.minimisationFunction,
                self.initial_guess,
                method=self.short_name,
                bounds=self.objective_function.boundaries,
                options=options);

        else:
            result = optimize.minimize(self.objective_function.minimisationFunction,
                self.initial_guess,
                method=self.short_name,
                bounds=self.objective_function.boundaries,
                jac='2-point',
                options=options);

        self.best_solution = Solution(self.objective_function, 1, result.x)
