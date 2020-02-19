import Particle as PART

from Optimiser import *

NoneType = type(None);


class PSO(Optimiser):

    def __init__(self, aCostFunction, aNumberOfParticles, initial_guess = None):

        super().__init__(aCostFunction, initial_guess);

        # Name of the algorithm
        self.full_name = "Particle Swarm Optimisation";
        self.short_name = "PSO";
        self.number_created_children = 0;

        # Add initial guess if any
        if not isinstance(self.initial_guess, NoneType):
            self.current_solution_set.append(PART.Particle(
                self.objective_function,
                self,
                self.initial_guess));

        # Create the swarm
        while (self.getNumberOfParticles() < aNumberOfParticles):
            self.current_solution_set.append(PART.Particle(self.objective_function, self, self.objective_function.initialRandomGuess()));

        # Number of new particles created
        self.number_created_particles = self.getNumberOfParticles();

        # Number of new particles moved
        self.number_moved_particles = 0;

        # Store the best particle
        self.best_solution = None;
        self.average_objective_value = None;
        self.saveBestParticle();



    def evaluate(self, aParameterSet):
        return self.objective_function.evaluate(aParameterSet, 1);

    def getNumberOfParticles(self):
        return len(self.current_solution_set);

    def saveBestParticle(self):
        # Compute the objective value of all the particles
        # And keep track of who is the best particles
        best_particle_index = 0;

        self.average_objective_value = 0;

        for i in range(self.getNumberOfParticles()):
            self.average_objective_value += self.current_solution_set[i].getObjective();

            if (self.current_solution_set[best_particle_index].getObjective() > self.current_solution_set[i].getObjective()):
                best_particle_index = i;

        self.average_objective_value /= self.getNumberOfParticles();

        if isinstance(self.best_solution, NoneType):
            self.best_solution =  self.current_solution_set[best_particle_index].copy();
        elif self.best_solution.getObjective() > self.current_solution_set[best_particle_index].getObjective():
            self.best_solution =  self.current_solution_set[best_particle_index].copy();

    def runIteration(self):

        # For each particle
        for particle in self.current_solution_set:

            # Update the particle's position and velocity
            particle.update();

        # Update the number of particles moved
        self.number_moved_particles += self.getNumberOfParticles();

        # Update the swarm's best known position
        self.saveBestParticle()

        # Return the best individual
        return self.best_solution;



    def __repr__(self):
        value = ""

        for particle in self.current_solution_set:
            value += particle.__repr__();
            value += '\n';

        return value;
