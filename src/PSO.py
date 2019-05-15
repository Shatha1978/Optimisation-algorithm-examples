from Optimiser import *
from Particle import *

class PSO(Optimiser):

    def __init__(self, aNumberOfDimensions, aBoundarySet, aCostFunction, aNumberOfParticles):

        super().__init__(aBoundarySet, aCostFunction);

        # Save the best particle at each iteration
        #self.best_solution_set = [];

        # and copy input parameters
        self.number_of_dimensions = aNumberOfDimensions;

        # Add a particle
        self.current_solution_set.append(Particle(aNumberOfDimensions, aBoundarySet, aCostFunction, self))

        # Keep track of the best particle
        best_particle_index = 0;

        # Create the particles
        while len(self.current_solution_set) < aNumberOfParticles:
            self.current_solution_set.append(Particle(aNumberOfDimensions, aBoundarySet, aCostFunction, self))

            # The new particle is better
            # Minimisation
            if self.current_solution_set[best_particle_index].cost > self.current_solution_set[-1].cost:
                best_particle_index = len(self.current_solution_set) - 1;

        # Store the best particle
        #self.best_solution_set.append(copy.deepcopy(best_particle))
        self.best_solution = self.current_solution_set[best_particle_index].copy();

    def run(self):

        # Keep track of the best particle
        best_cost = self.best_solution.cost;
        best_particle_index = -1;

        # For each partical
        for i in range(len(self.current_solution_set)):

            # update the particales' positions and velocities
            self.current_solution_set[i].update()

            # The new particle is better
            if best_cost > self.current_solution_set[i].cost:
                best_cost = self.current_solution_set[i].cost
                best_particle_index = i;

        # The new particle is better
        if best_particle_index != -1:
            self.best_solution = self.current_solution_set[best_particle_index].copy();

        return self.best_solution;
        #return self.best_solution_set[-1];



    def __repr__(self):
        value = ""

        for particle in self.current_solution_set:
            value += particle.__repr__();
            value += '\n';

        return value;
