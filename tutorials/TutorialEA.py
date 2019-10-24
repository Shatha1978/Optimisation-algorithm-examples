#!/usr/bin/env python3

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

# Import objective function
from TestProblem         import *

# Create test problem
test_problem = TestProblem();


# Parameters for EA
number_of_individuals            = 10;
number_of_generation             = 10;

# Create the optimiser
optimiser = EvolutionaryAlgorithm(test_problem,
    number_of_individuals);


print ("Initial best individual: ", optimiser.best_solution)

# Set the selection operator
#optimiser.setSelectionOperator(TournamentSelection(3));
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

for i in range(1, number_of_generation):
    print ("Run Generation ", i, "/", number_of_generation)
    optimiser.runIteration();
    print ("Best individual: ", optimiser.best_solution)
