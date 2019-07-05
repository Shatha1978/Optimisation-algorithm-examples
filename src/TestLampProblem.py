#!/usr/bin/env python3

from LampFunction import *

import Comparison

# Stopping criteria
max_iterations = 100;

# Instantiate the objective function
test_problem = LampGlobalFitnessFunction(200, 100, 50, 10);

# Number of runs
number_of_runs = 15;


def callback(optimiser, file_prefix, run_id):
    optimiser.objective_function.saveImage(optimiser.best_solution, file_prefix + optimiser.short_name + "_" + str(run_id) + ".txt");



Comparison.run(test_problem, max_iterations, number_of_runs, "lamp_", False, callback);
