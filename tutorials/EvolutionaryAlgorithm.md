

```python
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

# Parameters for EA
g_number_of_individuals            = 100;
g_number_of_generation             = 100;

# Create the optimiser
optimiser = EvolutionaryAlgorithm(g_test_problem,
    g_number_of_individuals,
    initial_guess=initial_guess);

# Set the selection operator
#optimiser.setSelectionOperator(TournamentSelection(3));
#optimiser.setSelectionOperator(RouletteWheel());
optimiser.setSelectionOperator(RankSelection());

# Create the genetic operators
elitism = ElitismOperator(0.1);
new_blood = NewBloodOperator(0.1);
blend_cross_over = BlendCrossoverOperator(0.6, gaussian_mutation);

# Add the genetic operators to the EA
optimiser.addGeneticOperator(new_blood);
optimiser.addGeneticOperator(gaussian_mutation);
optimiser.addGeneticOperator(blend_cross_over);
optimiser.addGeneticOperator(elitism);

for _ in range(1, g_number_of_generation):
    optimiser.runIteration();

# Get the best individual
...

```
Select the selection operator

```python
class SelectionOperator:

    # Constructor
    # name: name of the selection operator
    def __init__(self, name: str="Unspecified selection operator"):
        self.name = name;

    # Accessor on the name of the operator
    def getName(self) -> str:
        return self.name;

    # Select a good individual from anIndividualSet
    def select(self, anIndividualSet):
        return self.selectGood(anIndividualSet);

    # Select a good individual from anIndividualSet
    # Useful for a steady-state EA
    def selectGood(self, anIndividualSet):
        return self.__select__(anIndividualSet, True);

    # Select a bad individual from anIndividualSet
    # Useful for a steady-state EA
    def selectBad(self, anIndividualSet):
        return self.__select__(anIndividualSet, False);

    # Run this method once per generation, before any selection is done. Useful for ranking the individuals
    def preProcess(self, anIndividualSet):
        raise NotImplementedError("Subclasses should implement this!")

    # Abstract method to perform the actual selection
    def __select__(self, anIndividualSet, aFlag): # aFlag == True for selecting good individuals,
                                                  # aFlag == False for selecting bad individuals,
        raise NotImplementedError("Subclasses should implement this!")

    # Method used for print()
    def __str__(self) -> str:
        return "name:\t\"" + self.name + "\"";

```
