#!/usr/bin/env python3

from scipy import optimize

from AckleyFunction import *

import pandas as pd

from PSO import *
from SimulatedAnnealing import *
from EvolutionaryAlgorithm import *

# Selection operators
from TournamentSelection      import *
from RouletteWheel            import *
from RankSelection            import *

# Genetic operators
from ElitismOperator          import *
from BlendCrossoverOperator   import *
from GaussianMutationOperator import *
from NewBloodOperator         import *

# Store the results for each optimisation method
columns = ['Methods','x','y','Euclidean distance', 'Number of evaluations'];
df = pd.DataFrame (columns = columns);

# Stopping criteria
max_iterations = 100;

# Instantiate the objective function
test_problem = AckleyFunction(2);

# Create a random guess common to all the optimisation methods
initial_guess = test_problem.initialRandomGuess();

# Optimisation methods implemented in scipy.optimize
methods = ['Nelder-Mead',
    'Powell',
    'CG',
    'BFGS',
    'L-BFGS-B',
    'TNC',
    'COBYLA',
    'SLSQP'
];

for method in methods:
    test_problem.number_of_evaluation = 0;

    # Methods that cannot handle constraints or bounds.
    if method == 'Nelder-Mead' or method == 'Powell' or method == 'CG' or method == 'BFGS' or method == 'COBYLA':

        result = optimize.minimize(test_problem.minimisationFunction,
            initial_guess,
            method=method,
            options={'maxiter': max_iterations});

    elif method == 'L-BFGS-B' or method == 'TNC' or method == 'SLSQP':
        result = optimize.minimize(test_problem.minimisationFunction,
            initial_guess,
            method=method,
            bounds=test_problem.boundaries,
            options={'maxiter': max_iterations});

    else:
        result = optimize.minimize(test_problem.minimisationFunction,
            initial_guess,
            method=method,
            bounds=test_problem.boundaries,
            jac='2-point',
            options={'maxiter': max_iterations});

    data = [[method, result.x[0], result.x[1], test_problem.getDistanceToGlobalOptimum(result.x), test_problem.number_of_evaluation]];

    df = df.append(pd.DataFrame(data, columns = columns));


# Parameters for EA
g_number_of_individuals            = 100;
g_iterations = int(max_iterations / g_number_of_individuals) + 1;

g_max_mutation_sigma = 0.1;
g_min_mutation_sigma = 0.01;

g_current_sigma = g_max_mutation_sigma;

def visualisationCallback():
    global g_current_sigma;

    # Update the mutation variance so that it varies linearly from g_max_mutation_sigma to
    # g_min_mutation_sigma
    if g_iterations > 1:
        g_current_sigma -= (g_max_mutation_sigma - g_min_mutation_sigma) / (g_iterations - 1);

    # Make sure the mutation variance is up-to-date
    gaussian_mutation.setMutationVariance(g_current_sigma);



# Optimisation and visualisation
optimiser = EvolutionaryAlgorithm(test_problem, g_number_of_individuals)

# Set the selection operator
#optimiser.setSelectionOperator(TournamentSelection(2));
#optimiser.setSelectionOperator(RouletteWheel());
optimiser.setSelectionOperator(RankSelection());

# Create the genetic operators
elitism = ElitismOperator(0.1);
new_blood = NewBloodOperator(0.1);
gaussian_mutation = GaussianMutationOperator(0.1, 0.2);
blend_cross_over = BlendCrossoverOperator(0.6, gaussian_mutation);

# Add the genetic operators to the EA
optimiser.addGeneticOperator(new_blood);
optimiser.addGeneticOperator(gaussian_mutation);
optimiser.addGeneticOperator(blend_cross_over);
optimiser.addGeneticOperator(elitism);

test_problem.number_of_evaluation = 0;
optimiser.plotAnimation(g_iterations, visualisationCallback);
data = [["EA", optimiser.best_solution.genes[0], optimiser.best_solution.genes[1], test_problem.getDistanceToGlobalOptimum(optimiser.best_solution.genes), test_problem.number_of_evaluation]];
df = df.append(pd.DataFrame(data, columns = columns));

# Optimisation and visualisation
test_problem.number_of_evaluation = 0;
optimiser = PSO(test_problem, g_number_of_individuals);
optimiser.plotAnimation(g_iterations);
data = [["PSO", optimiser.best_solution.position[0], optimiser.best_solution.position[1], test_problem.getDistanceToGlobalOptimum(optimiser.best_solution.position), test_problem.number_of_evaluation]];
df = df.append(pd.DataFrame(data, columns = columns));


# Optimisation and visualisation
test_problem.number_of_evaluation = 0;
optimiser = SimulatedAnnealing(test_problem, 5000, 0.04);
optimiser.plotAnimation(211);
data = [["SA", optimiser.best_solution.parameter_set[0], optimiser.best_solution.parameter_set[1], test_problem.getDistanceToGlobalOptimum(optimiser.best_solution.parameter_set), test_problem.number_of_evaluation]];
df = df.append(pd.DataFrame(data, columns = columns));

print(df)
