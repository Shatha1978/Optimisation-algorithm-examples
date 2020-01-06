#!/usr/bin/env python3

# Add a progress bar
from progress.bar import IncrementalBar

from EvolutionaryAlgorithm import *

# Selection operators
from TournamentSelection      import *
from RouletteWheelSelection   import *
from RankSelection            import *

# Genetic operators
from ElitismOperator          import *
from BlendCrossoverOperator   import *
from GaussianMutationOperator import *
from NewBloodOperator         import *

# Import objective function
from TestProblem         import *

# Create test problem
test_problem = TestProblem();


# Parameters for EA
number_of_individuals            = 50;
number_of_generation             = 50;

# Create the optimiser
optimiser = EvolutionaryAlgorithm(test_problem,
    number_of_individuals);


print ("Initial best individual: ", optimiser.best_solution)

# Set the selection operator
#optimiser.setSelectionOperator(TournamentSelection(3));
#optimiser.setSelectionOperator(RouletteWheelSelection());
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

# Run the evolutionary loop
bar = IncrementalBar('Generation', max=number_of_generation, suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
for i in range(number_of_generation):
    optimiser.runIteration();
    bar.next();
bar.finish();

# Get the final answer
parameters, objective = optimiser.getBestSolution();
print ("Problem solution: ", parameters);

# Get the fitness function
print ("Fitness function: ", objective);

# Get the Euclidean distance to the global optimum
print ("Euclidean distance to the global optimum: ", test_problem.getEuclideanDistanceToGlobalOptimum(parameters));
