import copy; # For deepcopy

class Solution:

    # aFlag: 1 for minimisation, 2 for maximisation (default value: 0)

    def __init__(self, anObjectiveFunction, aFlag, aParameterSet, aComputeObjectiveFlag = True):

        # Store the class attributes
        self.objective_function = anObjectiveFunction;
        self.flag = aFlag;
        self.parameter_set = [];

        # Initialise the objective value
        if self.flag == 1: # Minimisation
            self.objective = float('inf');
        elif self.flag == 2: # Maximisation
            self.objective = -float('inf');
        else: # Unknown
            self.objective = 0;

        # Copy the parameters if any
        if not isinstance(aParameterSet, (str, type(None))):
            self.parameter_set = copy.deepcopy(aParameterSet);

            if aComputeObjectiveFlag:
                self.computeObjectiveFunction();

    def copy(self):
        temp = Solution(self.objective_function, self.flag, self.parameter_set, False);
        temp.objective = self.objective;
        return temp;

    def computeObjectiveFunction(self):

        # Compute the fitness function
        self.objective = self.objective_function.evaluate(self.parameter_set, self.flag);

        return self.objective;

    def getParameter(self, i):
        if i >= len(self.parameter_set):
            raise IndexError;
        else:
            return self.parameter_set[i];

    def getObjective(self):
        return self.objective;

    def __repr__(self):
        value = "Parameters: [";

        for param in self.parameter_set:
            value += str(param);
            value += ',';
            
        value += "]\tFlag: ";
        value += str(self.flag);
        value += "\tObjective: ";
        value += str(self.getObjective());
        return value;
