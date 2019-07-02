#!/usr/bin/env python3

from scipy import optimize

from AckleyFunction import *

import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

from PSO import *
from SimulatedAnnealing import *
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

# Store the results for each optimisation method
columns = ['Run', 'Methods','x','y','Euclidean distance', 'Evaluations'];
df = pd.DataFrame (columns = columns);

# Stopping criteria
max_iterations = 500;

# Instantiate the objective function
test_problem = AckleyFunction(2);

number_of_runs = 15;


def cooling():
    global initial_temperature;
    global cooling_rate;
    global test_problem;

    return initial_temperature * math.pow(cooling_rate, test_problem.number_of_evaluation);


for run in range(number_of_runs):

    # Create a random guess common to all the optimisation methods
    initial_guess = test_problem.initialRandomGuess();

    # Optimisation methods implemented in scipy.optimize
    methods = ['Nelder-Mead',
        'Powell',
        'CG',
        'BFGS',
        'L-BFGS-B',
        'TNC',
        'COBYLA',
        'SLSQP'
    ];

    for method in methods:
        test_problem.number_of_evaluation = 0;

        # Methods that cannot handle constraints or bounds.
        if method == 'Nelder-Mead' or method == 'Powell' or method == 'CG' or method == 'BFGS' or method == 'COBYLA':

            result = optimize.minimize(test_problem.minimisationFunction,
                initial_guess,
                method=method,
                options={'maxiter': max_iterations});

        elif method == 'L-BFGS-B' or method == 'TNC' or method == 'SLSQP':
            result = optimize.minimize(test_problem.minimisationFunction,
                initial_guess,
                method=method,
                bounds=test_problem.boundaries,
                options={'maxiter': max_iterations});

        else:
            result = optimize.minimize(test_problem.minimisationFunction,
                initial_guess,
                method=method,
                bounds=test_problem.boundaries,
                jac='2-point',
                options={'maxiter': max_iterations});

        data = [[run, method, result.x[0], result.x[1], test_problem.getDistanceToGlobalOptimum(result.x), test_problem.number_of_evaluation]];

        df = df.append(pd.DataFrame(data, columns = columns));


    # Parameters for EA
    g_number_of_individuals            = 50;
    g_iterations = int(max_iterations / g_number_of_individuals);

    g_max_mutation_sigma = 0.1;
    g_min_mutation_sigma = 0.01;

    g_current_sigma = g_max_mutation_sigma;

    def visualisationCallback():
        global g_current_sigma;

        # Update the mutation variance so that it varies linearly from g_max_mutation_sigma to
        # g_min_mutation_sigma
        if g_iterations > 1:
            g_current_sigma -= (g_max_mutation_sigma - g_min_mutation_sigma) / (g_iterations - 1);

        # Make sure the mutation variance is up-to-date
        gaussian_mutation.setMutationVariance(g_current_sigma);



    # Optimisation and visualisation
    optimiser = EvolutionaryAlgorithm(test_problem, g_number_of_individuals)

    # Set the selection operator
    #optimiser.setSelectionOperator(TournamentSelection(2));
    #optimiser.setSelectionOperator(RouletteWheel());
    optimiser.setSelectionOperator(RankSelection());

    # Create the genetic operators
    elitism = ElitismOperator(0.1);
    new_blood = NewBloodOperator(0.1);
    gaussian_mutation = GaussianMutationOperator(0.1, 0.2);
    blend_cross_over = BlendCrossoverOperator(0.6, gaussian_mutation);

    # Add the genetic operators to the EA
    optimiser.addGeneticOperator(new_blood);
    optimiser.addGeneticOperator(gaussian_mutation);
    optimiser.addGeneticOperator(blend_cross_over);
    optimiser.addGeneticOperator(elitism);

    test_problem.number_of_evaluation = 0;

    if run == 0:
        optimiser.plotAnimation(g_iterations, visualisationCallback);
    else:
        for i in range(g_iterations):
            optimiser.runIteration();
            visualisationCallback();

    data = [[run, "EA", optimiser.best_solution.genes[0], optimiser.best_solution.genes[1], test_problem.getDistanceToGlobalOptimum(optimiser.best_solution.genes), test_problem.number_of_evaluation]];
    df = df.append(pd.DataFrame(data, columns = columns));

    # Optimisation and visualisation
    test_problem.number_of_evaluation = 0;
    optimiser = PSO(test_problem, g_number_of_individuals);

    if run == 0:
        optimiser.plotAnimation(g_iterations);
    else:
        for i in range(g_iterations - 1):
            optimiser.runIteration();

    data = [[run, "PSO", optimiser.best_solution.position[0], optimiser.best_solution.position[1], test_problem.getDistanceToGlobalOptimum(optimiser.best_solution.position), test_problem.number_of_evaluation]];
    df = df.append(pd.DataFrame(data, columns = columns));


    # Optimisation and visualisation
    test_problem.number_of_evaluation = 0;
    initial_temperature = 50000;
    cooling_rate = 0.98;

    optimiser = SimulatedAnnealing(test_problem, 5000, 0.04);
    optimiser.cooling_schedule = cooling;

    if run == 0:
        optimiser.plotAnimation(max_iterations);
    else:
        for t in range(max_iterations):
            optimiser.runIteration();
        print(optimiser.current_temperature)

    data = [[run, "SA", optimiser.best_solution.parameter_set[0], optimiser.best_solution.parameter_set[1], test_problem.getDistanceToGlobalOptimum(optimiser.best_solution.parameter_set), test_problem.number_of_evaluation]];
    df = df.append(pd.DataFrame(data, columns = columns));

    df.to_csv ('summary.csv', index = None, header=True)


def boxplot(column, title, filename, sort):

    plt.figure();

    # use dict comprehension to create new dataframe from the iterable groupby object
    # each group name becomes a column in the new dataframe
    df2 = pd.DataFrame({col:vals[column] for col, vals in df.groupby('Methods')})

    # find and sort the median values in this new dataframe
    #meds = df2.mean().sort_values(ascending=sort)

    # use the columns in the dataframe, ordered sorted by median value
    # return axes so changes can be made outside the function
    #df2[meds.index].boxplot(return_type="axes")
    #df2[['EA', 'SA', 'PRS', 'Nelder-Mead', 'CG']].boxplot(return_type="axes")
    df2.boxplot(return_type="axes")

    plt.title(title)
    plt.suptitle("")
    plt.xlabel('Optimisation method');
    plt.tight_layout()
    plt.autoscale()
    fig = plt.gcf()
    fig.set_size_inches(17.5, 10.5)
    plt.savefig(filename, orientation='landscape', bbox_inches = "tight")


boxplot('Evaluations',      'Number of evaluations',      'evaluations.pdf', False)

boxplot('Euclidean distance',      'Euclidean distance',      'distance.pdf', False)
