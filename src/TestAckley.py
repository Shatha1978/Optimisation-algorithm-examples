#!/usr/bin/env python3

from scipy import optimize

from AckleyFunction import *

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


test_problem = AckleyFunction();

methods = ['Nelder-Mead',
    'Powell',
    'CG',
    'BFGS',
    'L-BFGS-B',
    'TNC',
    'COBYLA',
    'SLSQP'
];

number_of_evaluation_set = [];
solution_set = [];
initial_guess = test_problem.initialGuess();

max_iterations = 0;

for method in methods:
    test_problem.number_of_evaluation = 0;

    print ("Run ", method);
    test_problem.number_of_evaluation = 0;
    test_problem.global_fitness = -float('inf');

    if method == 'Nelder-Mead' or method == 'CG' or method == 'Powell' or method == 'BFGS' or method == 'L-BFGS-B' or method == 'TNC' or method == 'COBYLA' or method == 'SLSQP':
        result = optimize.minimize(test_problem.minimisationFunction,
            initial_guess,
            method=method,
            bounds=test_problem.boundaries);
    else:
        result = optimize.minimize(test_problem.minimisationFunction,
            initial_guess,
            method=method,
            bounds=test_problem.boundaries,
            jac='2-point');

    if max_iterations < test_problem.number_of_evaluation:
        max_iterations = test_problem.number_of_evaluation;

    print(method, ".number_of_evaluation:\t", test_problem.number_of_evaluation)

    number_of_evaluation_set.append(test_problem.number_of_evaluation);
    solution_set.append(result.x);


# Parameters for EA
g_iterations            = 100;
g_number_of_individuals = int(max_iterations / g_iterations) + 1;

g_max_mutation_sigma = 0.1;
g_min_mutation_sigma = 0.01;

g_current_sigma = g_max_mutation_sigma;

def visualisationCallback():
    global g_current_sigma;

    # Update the mutation variance so that it varies linearly from g_max_mutation_sigma to
    # g_min_mutation_sigma
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
EA_number_of_evaluation = test_problem.number_of_evaluation
EA_solution = optimiser.best_solution;

# Optimisation and visualisation
test_problem.number_of_evaluation = 0;
optimiser = PSO(test_problem, g_number_of_individuals);
optimiser.plotAnimation(g_iterations);
PSO_number_of_evaluation = test_problem.number_of_evaluation
PSO_solution = optimiser.best_solution;


# Optimisation and visualisation
test_problem.number_of_evaluation = 0;
optimiser = SimulatedAnnealing(test_problem, 5000, 0.04);
optimiser.plotAnimation(211);
SA_number_of_evaluation = test_problem.number_of_evaluation
SA_solution = optimiser.best_solution;


for method, number_of_evaluation, solution in zip(methods, number_of_evaluation_set, solution_set):
    print(method, ".number_of_evaluation:\t", number_of_evaluation)
    print(method, " solution:\t", solution, "\tdistance:\t", test_problem.getDistanceToGlobalOptimum(solution));
    print()


print("EA.number_of_evaluation:\t", EA_number_of_evaluation)
print("EA solution:\t", EA_solution, "\tdistance:\t", test_problem.getDistanceToGlobalOptimum(EA_solution.genes));
print()
print("PSO.number_of_evaluation:\t", PSO_number_of_evaluation)
print("PSO solution:\t", PSO_solution, "\tdistance:\t", test_problem.getDistanceToGlobalOptimum(PSO_solution.position));
print()
print("SA.number_of_evaluation:\t", SA_number_of_evaluation)
print("SA solution:\t", SA_solution, "\tdistance:\t", test_problem.getDistanceToGlobalOptimum(SA_solution.parameter_set));
print()
