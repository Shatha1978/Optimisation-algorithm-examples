import copy; # For deepcopy

class Solution:
    def __init__(self, aParameterSet = None):

        self.parameter_set = [];

        if aParameterSet != None:
            self.parameter_set = copy.deepcopy(aParameterSet);

        self.energy = float('inf');

    def getParameter(self, i):
        if i >= len(self.parameter_set):
            raise IndexError;
        else:
            return self.parameter_set[i];

    def getObjective(self):
        return self.energy;

    def __repr__(self):
        value = "Parameters: ";
        value += ' '.join(str(e) for e in self.parameter_set)
        value += "\tEnergy: ";
        value += str(self.energy);
        return value;
