"""@package SimulatedAnnealing
This package implements the simulated annealing (SA) optimisation method. SA is a metaheuristic to find the global optimum in an optimization problem.
For details, see https://en.wikipedia.org/wiki/Simulated_annealing and  Kirkpatrick, S.; Gelatt Jr, C. D.; Vecchi, M. P. (1983). "Optimization by Simulated Annealing". Science. 220 (4598): 671–680. doi:10.1126/science.220.4598.671.
@author Dr Franck P. Vidal, Bangor University
@date 15th May 2019
"""

#################################################
# import packages
###################################################
import math; # For exp
import copy; # For deepcopy
import random; # For uniform

## \class This class implements the simulated annealing optimisation method
class SimulatedAnnealing:

    ## \brief Constructor.
    # \param self
    # \param aNumberOfDimensions: The number of dimensions of the search space
    # \param aBoundarySet: For each dimension, the range of possible values
    # \param aCostFunction: The cost function to minimise
    # \param aTemperature: The initial temperature of the system (default value: 10,000)
    # \param aCoolingRate: The cooling rate (i.e. how fast the temperature will decrease) (default value: 0.003)
    def __init__(self, aNumberOfDimensions, aBoundarySet, aCostFunction, aTemperature = 10000, aCoolingRate = 0.003):

        # Initialise attributes
        self.initStates()

        # and copy input parameters
        self.number_of_dimensions = aNumberOfDimensions;
        self.boundary_set = copy.deepcopy(aBoundarySet);
        self.cost_function = aCostFunction;
        self.initial_temperature = aTemperature;
        self.cooling_rate = aCoolingRate;

        # Create the current solution from random
        self.current_solution = [];
        for i in range(aNumberOfDimensions):
            self.current_solution.append(random.uniform(self.boundary_set[i][0], self.boundary_set[i][1]));

    ## \brief Initialise attributes.
    # \param self
    def initStates(self):
        self.best_energy = float("inf");
        self.best_solution = [];

        self.min_energy =  float("inf");
        self.max_energy = -float("inf");

        self.temperature_set = [];

        self.current_solution_set = [];
        self.best_solution_set = [];

        self.current_energy_set = [];
        self.best_energy_set = [];

    ## \brief Compute the energy corresponding to a given solution.
    # \param self
    # \param aSolution: The solution to assess
    # \return the corresponding energy
    def computeEnergy(self, aSolution):
        # Compute the cost function
        energy = self.cost_function(aSolution);

        # Keep track of the min/max cost values
        self.min_energy = min(self.min_energy, energy);
        self.max_energy = max(self.max_energy, energy);

        # Return the corresponding cost
        return energy;

    ## \brief Compute the acceptance probability corresponding to an energy.
    # \param self
    # \param aNewEnergy: The energy to assess
    # \return the corresponding acceptance probability
    def acceptanceProbability(self, aNewEnergy):

        # The new soluation is better (lower energy), keep it
        if aNewEnergy < self.current_energy:
            return 1.0;
        # The new soluation is worse, calculate an acceptance probability
        else:
            return math.exp((self.current_energy - aNewEnergy) / self.current_temperature);

    ## \brief Get a neighbour in the vicinity of a given solution.
    # \param self
    # \param aSolution: The solution to assess
    # \return a neighbour
    def getRandomNeighbour(self, aSolution):
        # Start with an empty solution
        new_solution = [];

        # Process each dimension of the search space
        for i in range(self.number_of_dimensions):
            min_val = self.boundary_set[i][0];
            max_val = self.boundary_set[i][1];
            range_val = max_val - min_val;
            new_solution.append(random.gauss(min_val + range_val / 2, range_val * 0.1));

        return (copy.deepcopy(new_solution));

    ## \brief Get a neighbour in the vicinity of a given solution.
    # \param self
    # \param aSolution: The solution to assess
    # \return a neighbour
    def getRandomNeighbor(self, aSolution):
        return self.getRandomNeighbour(aSolution);

    ## \brief Run the optimisation.
    # \param self
    # \param aRetartFlag: True if the algorithm has to run twice, False if it has to run only once (default value: False)
    # \param aVerboseFlag: True if intermediate results are printing in the terminal, False to print no intermediate results (default value: False)
    def run(self, aRetartFlag = False, aVerboseFlag = False):

        self.current_temperature = self.initial_temperature;

        self.initStates();

        # Compute its energy using the cost function
        self.current_energy = self.computeEnergy(self.current_solution);

        # This is also the best solution so far
        self.best_solution = copy.deepcopy(self.current_solution);
        self.best_energy = self.current_energy;

        iteration = 0;
        if aVerboseFlag:
            header  = "iteration";
            header += " temperature";
            for i in range(self.number_of_dimensions):
                header += " best_solution[" + str(i) + "]";
            header += " best_solution_energy";
            for i in range(self.number_of_dimensions):
                header += " current_solution[" + str(i) + "]";
            header += " current_solution_energy";
            print(header);
            print(self.iterationDetails(iteration));

        self.temperature_set.append(self.current_temperature);
        self.current_solution_set.append(self.current_solution);
        self.best_solution_set.append(self.best_solution);
        self.current_energy_set.append(self.current_energy);
        self.best_energy_set.append(self.best_energy);

        # Loop until system has cooled
        while self.current_temperature > 1.0:

            if aRetartFlag:
                if iteration != 0:
                    if (self.current_energy - self.min_energy) / (self.max_energy - self.min_energy) > 0.9:
                        #print("Restart")
                        self.current_solution = self.best_solution;
                        self.current_energy   = self.best_energy;

            # Create a new solution depending on the current solution,
            # i.e. a neighbour
            neighbour = self.getRandomNeighbour(self.current_solution);

            # Get its energy (cost function)
            neighbour_energy = self.computeEnergy(neighbour);

            # Accept the neighbour or not depending on the acceptance probability
            if self.acceptanceProbability(neighbour_energy) > random.uniform(0, 1):
                self.current_solution = copy.deepcopy(neighbour);
                self.current_energy = neighbour_energy;

            # The neighbour is better thant the current element
            if self.best_energy > self.current_energy:
                #print("Best energy was ", self.best_energy, "it is now ", self.current_energy)
                self.best_solution = copy.deepcopy(self.current_solution);
                self.best_energy = self.current_energy;

            iteration = iteration + 1;

            if aVerboseFlag:
                print(self.iterationDetails(iteration));

            self.temperature_set.append(self.current_temperature);
            self.current_solution_set.append(self.current_solution);
            self.best_solution_set.append(self.best_solution);
            self.current_energy_set.append(self.current_energy);
            self.best_energy_set.append(self.best_energy);

            # Cool the system
            self.current_temperature *= 1.0 - self.cooling_rate;

        self.current_solution = copy.deepcopy(self.best_solution);
        self.current_energy = self.best_energy;

    ## \brief Print the current solution and the best solution so far.
    # \param self
    # \return a string that includes the current solution and the best solution so far (parameters and corresponding costs)
    def iterationDetails(self, iteration):
        return (str(iteration) + ', ' +
            str(self.current_temperature) + ', ' +
            ' '.join(str(e) for e in self.best_solution) + ', ' +
            str(self.best_energy) + ', ' +
            ' '.join(str(e) for e in self.current_solution) + ', ' +
            str(self.current_energy));

    ## \brief Print the best solution.
    # \param self
    # \return a string that includes the best solution parameters and its corresponding cost
    def __repr__(self):
        value = "Best solution: ";
        value += ' '.join(str(e) for e in self.best_solution)
        value += "\tCorresponding cost: ";
        value += str(self.best_energy);
        return value;
