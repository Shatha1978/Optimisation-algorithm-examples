import random
import numpy

from SelectionOperator import *

class ThresholdSelection(SelectionOperator):

    def __init__(self,
                 aThreshold,
                 anAlternativeSelectionOperator,
                 aMaxIteration = 50):

        super().__init__("Threshold selection");

        self.threshold = aThreshold;
        self.alternative_selection_operator = anAlternativeSelectionOperator;
        self.max_iteration = aMaxIteration;

        # Get a SystemRandom instance out of random package
        self.system_random = random.SystemRandom();

    def preProcess(self, anIndividualSet):
        return

    def __str__(self):
        return super().__str__() + "\t" + "tournament_size:\t" + str(self.tournament_size);

    def __select__(self, anIndividualSet, aFlag): # aFlag == True for selecting good individuals,
                                                  # aFlag == False for selecting bad individuals,

        max_ind = len(anIndividualSet) - 1;

        for i in range(self.max_iteration):
            selected_index = self.system_random.randint(0, max_ind)
            fitness = anIndividualSet[selected_index].computeObjectiveFunction()

            if aFlag == True:
                if fitness > self.threshold:
                    return selected_index;
            else:
                if fitness <= self.threshold:
                    return selected_index;

        return self.alternative_selection_operator.__select__(anIndividualSet, aFlag);
