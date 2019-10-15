#!/usr/bin/env python3

from AckleyFunction import *

import Comparison

# Stopping criteria
max_iterations = 200;

# Instantiate the objective function
test_problem = AckleyFunction(2);

# Number of runs
number_of_runs = 15;

Comparison.run(test_problem, max_iterations, number_of_runs, "ackley_", True);
