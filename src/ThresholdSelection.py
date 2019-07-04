# Import the random package to radomly select individuals
import random

# Import the superclass (also called base class), which is an abstract class,
# to implement the subclass ThresholdSelection
from SelectionOperator import *

# The subclass that inherits of SelectionOperator
class ThresholdSelection(SelectionOperator):

    # Constructor
    # aThreshold: the number of dimensions
    # anAlternativeSelectionOperator: when the threshold operator fails to find a suitable candidate in aMaxIteration, use anAlternativeSelectionOperator instead to select the individual
    # aMaxIteration: the max number of iterations
    def __init__(self,
                 aThreshold,
                 anAlternativeSelectionOperator,
                 aMaxIteration = 50):

        # Call the constructor of the superclass
        super().__init__("Threshold selection");

        # Store the attributes of the class
        self.threshold = aThreshold;
        self.alternative_selection_operator = anAlternativeSelectionOperator;
        self.max_iteration = aMaxIteration;

        # Get a SystemRandom instance out of random package
        self.system_random = random.SystemRandom();

    # Nothing to do
    def preProcess(self, anIndividualSet):
        return

    # Method used for print()
    def __str__(self):
        return super().__str__() + "\t" +
            "threshold:\t" + str(self.threshold) +
            "\tmax_iteration:\t" + str(self.max_iteration) +
            "\talternative selection operator:\t" + self.alternative_selection_operator;

    # Select an idividual
    # anIndividualSet: The set of individual to choose from
    # aFlag == True for selecting good individuals,
    # aFlag == False for selecting bad individuals,
    def __select__(self, anIndividualSet, aFlag):

        # The max individual ID
        max_ind = len(anIndividualSet) - 1;

        # Run the selection for a max of self.max_iteration times
        for _ in range(self.max_iteration):
            selected_index = self.system_random.randint(0, max_ind)
            fitness = anIndividualSet[selected_index].computeObjectiveFunction()

            # Try to find a good individual (candidate for reproduction)
            if aFlag == True:
                # The fitness is greater than the threshold, it's a good individual
                if fitness > self.threshold:
                    return selected_index;
            # Try to find a bad individual (candidate for death)
            else:
                # The fitness is lower than or equal to the threshold, it's a bad individual
                if fitness <= self.threshold:
                    return selected_index;

        # The threshold selection has failed self.max_iteration times,
        # use self.alternative_selection_operator instead
        return self.alternative_selection_operator.__select__(anIndividualSet, aFlag);
