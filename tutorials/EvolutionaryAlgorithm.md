It is assumed here that you completed the previous tutorial on how to define your objective function. See [TestProblem.py](TestProblem.py).

The main package for evolutionary algorithms is in [EvolutionaryAlgorithm.py](../src/EvolutionaryAlgorithm.py):
```python
from EvolutionaryAlgorithm import *
```

Popular selection mechanism are already implemented. The base class is in [SelectionOperator.py](SelectionOperator). It can be overloaded to implement actual operators. Rank selection, roulette wheel selection, and tournament selection operators are in:

- [RankSelection.py](../src/RankSelection.py),
- [RouletteWheelSelection.py](../src/RouletteWheelSelection.py),
- [TournamentSelection.py](../src/TournamentSelection.py).

Import the packages using:
```python
# Selection operators
from TournamentSelection      import *
from RouletteWheel            import *
from RankSelection            import *
```
Popular operator mechanism are already implemented. The base class is in [GeneticOperator.py](GeneticOperator). It can be overloaded to implement actual operators. Elitism, BlendCrossover, GaussianMutation and  NewBlood operators are in:

- [ElitismOperator.py](../src/ElitismOperator.py),
- [BlendCrossoverOperator.py](../src/BlendCrossoverOperator.py),
- [GaussianMutationOperator.py](../src/GaussianMutationOperator.py),
- [NewBloodOperator.py](../src/NewBloodOperator.py).

```python
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

# Run the evolutionary loop
from progress.bar import IncrementalBar
bar = IncrementalBar('Generation', max=number_of_generation, suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
for i in range(number_of_generation):
    optimiser.runIteration();
    bar.next();
bar.finish();

# Get the final answer
print ("Final answer: ", optimiser.best_solution);

# Get the fitness function
print ("Fitness function: ", optimiser.best_solution.objective);

# Get the Euclidean distance to the global optimum
print ("Euclidean distance to the global optimum: ", test_problem.getEuclideanDistanceToGlobalOptimum(parameters));
```


Get the source code of the tutorial:

- [TestProblem.py](TestProblem.py)
- [TutorialEA.py](TutorialEA.py)
