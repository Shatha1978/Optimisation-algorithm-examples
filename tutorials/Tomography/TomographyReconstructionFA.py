#!/usr/bin/env python3

import sys, os
import argparse

import math

import numpy as np

import logging;

# Add a progress bar
from progress.bar import IncrementalBar

import matplotlib.pyplot as plt

from skimage.io import imread, imsave



from EvolutionaryAlgorithm import *

# Selection operators
from TournamentSelection      import *
from RouletteWheelSelection   import *
from RankSelection            import *
from ThresholdSelection       import *

# Genetic operators
from ElitismOperator          import *
from BlendCrossoverOperator   import *
from GaussianMutationOperator import *
from NewBloodOperator         import *

from TomographyGlobalFitness import TomographyGlobalFitness
from TomographyLocalFitness  import TomographyLocalFitness

import ImageMetrics as IM;

# Import the circular list
from CircularList import *

import matplotlib
#matplotlib.use('PS')
matplotlib.use('QT5Agg')

NoneType = type(None);



# Check the command line arguments
def checkCommandLineArguments():
    global logging;
    global args;

    parser = argparse.ArgumentParser(description='Evolutionary reconstruction.')

    parser.add_argument('--input', help='Input image (groundtruth)',      nargs=1, type=str, required=True);

    parser.add_argument('--output_with_bad_flies', help='Reconstructed image with the bad flies',      nargs=1, type=str, required=False);

    parser.add_argument('--output_without_bad_flies', help='Reconstructed image without the bad flies',      nargs=1, type=str, required=False);

    parser.add_argument('--save_input_images', help='Where to save the input images (groundtruth with and without noise, and the sinogram)',      nargs=1, type=str, required=False);

    parser.add_argument('--angles', help='Number of angles',      nargs=1, type=int, required=True);

    parser.add_argument('--peak', help='Peak value for the Poisson noise',      nargs=1, type=float, required=False);

    parser.add_argument('--selection', help='Selection operator (threshold, tournament or dual)',      nargs=1, type=str, required=True);

    parser.add_argument('--initial_pop_size', help='Size of the initial population',      nargs=1, type=int, required=False);

    parser.add_argument('--tournament_size', help='Number of individuals involved in the tournament',      nargs=1, type=int, required=False, default=2);

    parser.add_argument('--generations', help='Number of generations',      nargs=1, type=int, required=True);

    parser.add_argument('--max_pop_size', help='Maximum number of individuals',      nargs=1, type=int, required=False);

    parser.add_argument('--steady_state', help='Realtime visualisation', action="store_true");

    parser.add_argument('--generational', help='Realtime visualisation', action="store_true");

    parser.add_argument('--visualisation', help='Realtime visualisation', action="store_true");

    parser.add_argument('--max_stagnation_counter', help='Max value of the stagnation counter to trigger a mitosis', nargs=1, type=int, required=True);

    parser.add_argument('--initial_lambda', help='Weight of the TV-norm regularisation at the start of the optimisation', nargs=1, type=float, required=True);

    parser.add_argument('--final_lambda', help='Weight of the TV-norm regularisation at the end of the optimisation', nargs=1, type=float, required=True);

    parser.add_argument('--initial_mutation_variance', help='Mutation variance at the start of the optimisation', nargs=1, type=float, required=True);

    parser.add_argument('--final_mutation_variance', help='Mutation variance at the end of the optimisation', nargs=1, type=float, required=True);

    parser.add_argument('--initial_new_blood_probability', help='Probability of the new blood operator at the start of the optimisation', nargs=1, type=float, required=True);

    parser.add_argument('--final_new_blood_probability', help='Probability of the new blood operator at the end of the optimisation', nargs=1, type=float, required=True);

    parser.add_argument('--logging', help='File name of the log file', nargs=1, type=str, required=False);

    parser.add_argument('--objective', help='Objective function: Valid values are: MAE, MSE, RMSE, NRMSE_euclidean, NRMSE_mean, NRMSE_min_max, cosine_similarity, mean_relative_error, max_relative_error, SSIM, PSNR, or ZNCC', nargs=1, type=str, required=True);

    args = parser.parse_args();

    # Set the logger if needed
    if not isinstance(args.logging, NoneType):
        logging.basicConfig(filename=args.logging[0],
                            level=logging.DEBUG,
                            filemode='w',
                            format='%(asctime)s, %(name)s - %(levelname)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')

        logging.debug(args)

    if args.steady_state and args.generational:
        raise ValueError('Options --steady_state and --generational can\'t be used at the same time. Choose one implementation.')

    if not args.steady_state and not args.generational:
        raise ValueError('Argument --steady_state or --generational should be used. Choose an implementation.')

    if not isinstance(args.objective, NoneType):

        if args.objective[0] not in IM.MINIMISATION and args.objective[0] not in IM.MAXIMISATION:
            raise ValueError('Argument --objective "%s" is not valid.' % args.objective[0])

    return args;

from scipy.stats import linregress

def getGradient(X, Y):
    gradient = 0;

    if x_set.len() > 1:
        #print(X.len(), Y.len())
        temp_X = np.array(X.values());
        temp_X = temp_X - temp_X.mean();

        temp_Y = np.array(Y.values());
        temp_Y = temp_Y - temp_Y.mean();

        gradient = np.dot(temp_X, temp_Y) / \
            np.square(temp_X).sum();

        #slope, intercept, r_value, p_value, std_err = linregress(X.values(), Y.values());
        #print("gradient", gradient)
        #print("slope", slope)
        #print("r_value", r_value)
        #print("p_value", p_value)
        #print("std_err", std_err)
        #print()
        #print()

    return gradient;


def filterBadIndividualsOut():
    number_of_good_individuals = 0;
    number_of_bad_individuals = 0;
    good_individual_set = [];

    if args.generational:
        for individual in optimiser.current_solution_set:
            if individual.getObjective() > 0:
                number_of_good_individuals += 1;
                for parameter in individual.parameter_set:
                    good_individual_set.append(parameter);
            else:
                number_of_bad_individuals += 1;

    elif args.steady_state:

        # Iteratively delete bad individuals
        number_of_bad_individuals = 1;
        while number_of_bad_individuals != 0:
            good_individual_set = [];

            number_of_bad_individuals = 0;
            number_of_good_individuals = 0;

            for i,j in zip(global_fitness_function.current_population[0::2], global_fitness_function.current_population[1::2]):

                if number_of_bad_individuals == 0:
                    local_fitness = local_fitness_function.objectiveFunction((i, j));

                    if local_fitness < 0.0:
                        number_of_bad_individuals += 1
                    else:
                        number_of_good_individuals += 1

                        good_individual_set.append(i);
                        good_individual_set.append(j);
                else:
                    good_individual_set.append(i);
                    good_individual_set.append(j);

            global_fitness_function.current_population = good_individual_set;
            global_fitness_function.objectiveFunction(good_individual_set, True);

    return number_of_good_individuals, \
        number_of_bad_individuals, \
        good_individual_set;


class MyBar(IncrementalBar):
    suffix = '%(index)d/%(max)d - %(percent).1f%% - %(eta)ds - Global fitness %(global_fitness)d - RMSE %(RMSE)d - TV %(TV)d - ZNCC %(zncc).1f%%'
    @property
    def global_fitness(self):
        global global_fitness_function;
        return global_fitness_function.global_fitness_set[-1]

    @property
    def RMSE(self):
        global global_fitness_function;
        return global_fitness_function.global_error_term_set[-1]

    @property
    def TV(self):
        global global_fitness_function;
        return global_fitness_function.global_regularisation_term_set[-1]

    @property
    def zncc(self):
        global global_fitness_function;
        return global_fitness_function.zncc_set[-1] * 100;


def linearInterpolation(start, end, i, j):
    return start + (end - start) * (1 - (j - i) / j);


g_first_log = True;
g_log_event = "";
g_generation = 0;
x_set  = CircularList(5);
global_fitness_set  = CircularList(5);
TV_set = CircularList(5);

def addToFitnessCircularList():
    global x_set, global_fitness_set, TV_set

    add_to_list = False;

    if len(global_fitness_function.global_fitness_set) > 0:
        if not isinstance(x_set.last_element, NoneType):
            if x_set.last_element != optimiser.number_created_children:
                add_to_list = True;
        else:
            add_to_list = True;

    if add_to_list:
        x_set.append(optimiser.number_created_children);
        global_fitness_set.append(global_fitness_function.global_fitness_set[-1]);
        TV_set.append(global_fitness_function.global_regularisation_term_set[-1]);

def logStatistics(aNumberOfIndividuals):

    global global_fitness_function;
    global g_first_log;
    global g_log_event;
    global g_generation;
    global optimiser;
    global x_set, global_fitness_set, TV_set;
    global last_proportion_of_good_flies;

    if not isinstance(args.logging, NoneType):
        if g_first_log:
            g_first_log = False;
            logging.info("generation,new_individual_counter,event,number_of_emission_points,MAE_sinogram,MSE_sinogram,RMSE_sinogram,NRMSE_euclidean_sinogram,NRMSE_mean_sinogram,NRMSE_min_max_sinogram,cosine_similarity_sinogram,SSIM_sinogram,PSNR_sinogram,ZNCC_sinogram,TV_sinogram,MAE_reconstruction,MSE_reconstruction,RMSE_reconstruction,NRMSE_euclidean_reconstruction,NRMSE_mean_reconstruction,NRMSE_min_max_reconstruction,cosine_similarity_reconstruction,SSIM_reconstruction,PSNR_reconstruction,ZNCC_reconstruction,TV_reconstruction,circular_list_mean,circular_list_std,max_iteration_reached_counter,gradient_global_fitness,gradient_TV,good_flies,bad_flies");

        ref  =  global_fitness_function.projections;
        test = global_fitness_function.population_sinogram_data;
        MAE_sinogram                 = IM.getMAE(ref, test);
        MSE_sinogram                 = IM.getMSE(ref, test);
        RMSE_sinogram                = IM.getRMSE(ref, test);
        NRMSE_euclidean_sinogram     = IM.getNRMSE_euclidean(ref, test);
        NRMSE_mean_sinogram          = IM.getNRMSE_mean(ref, test);
        NRMSE_min_max_sinogram       = IM.getNRMSE_minMax(ref, test);
        cosine_similarity_sinogram   = IM.getCosineSimilarity(ref, test);
        #mean_relative_error_sinogram = IM.getMeanRelativeError(ref, test);
        #max_relative_error_sinogram  = IM.getMaxRelativeError(ref, test);
        SSIM_sinogram                = IM.getSSIM(ref, test);
        PSNR_sinogram                = IM.getPSNR(ref, test);
        ZNCC_sinogram                = IM.getNCC(ref, test);
        TV_sinogram                  = IM.getTV(test);

        ref  =  global_fitness_function.image;
        test = global_fitness_function.population_image_data;
        MAE_reconstruction                 = IM.getMAE(ref, test);
        MSE_reconstruction                 = IM.getMSE(ref, test);
        RMSE_reconstruction                = IM.getRMSE(ref, test);
        NRMSE_euclidean_reconstruction     = IM.getNRMSE_euclidean(ref, test);
        NRMSE_mean_reconstruction          = IM.getNRMSE_mean(ref, test);
        NRMSE_min_max_reconstruction       = IM.getNRMSE_minMax(ref, test);
        cosine_similarity_reconstruction   = IM.getCosineSimilarity(ref, test);
        #mean_relative_error_reconstruction = IM.getMeanRelativeError(ref, test);
        #max_relative_error_reconstruction  = IM.getMaxRelativeError(ref, test);
        SSIM_reconstruction                = IM.getSSIM(ref, test);
        PSNR_reconstruction                = IM.getPSNR(ref, test);
        ZNCC_reconstruction                = IM.getNCC(ref, test);
        TV_reconstruction                  = IM.getTV(test);


        #logging.info("%i,%s,%i,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f" % (g_generation,g_log_event,MAE_sinogram,MSE_sinogram,RMSE_sinogram,NRMSE_euclidean_sinogram,NRMSE_mean_sinogram,NRMSE_min_max_sinogram,cosine_similarity_sinogram,mean_relative_error_sinogram,max_relative_error_sinogram,SSIM_sinogram,PSNR_sinogram,ZNCC_sinogram,TV_sinogram,MAE_reconstruction,MSE_reconstruction,RMSE_reconstruction,NRMSE_euclidean_reconstruction,NRMSE_mean_reconstruction,NRMSE_min_max_reconstruction,cosine_similarity_reconstruction,mean_relative_error_reconstruction,max_relative_error_reconstruction,SSIM_reconstruction,PSNR_reconstruction,ZNCC_reconstruction,TV_reconstruction));
        gradient_global_fitness = getGradient(x_set, global_fitness_set);
        gradient_TV             = getGradient(x_set, TV_set);

        logging.info("%i,%i,%s,%i,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f" % (g_generation,optimiser.number_created_children,g_log_event,aNumberOfIndividuals,MAE_sinogram,MSE_sinogram,RMSE_sinogram,NRMSE_euclidean_sinogram,NRMSE_mean_sinogram,NRMSE_min_max_sinogram,cosine_similarity_sinogram,SSIM_sinogram,PSNR_sinogram,ZNCC_sinogram,TV_sinogram,MAE_reconstruction,MSE_reconstruction,RMSE_reconstruction,NRMSE_euclidean_reconstruction,NRMSE_mean_reconstruction,NRMSE_min_max_reconstruction,cosine_similarity_reconstruction,SSIM_reconstruction,PSNR_reconstruction,ZNCC_reconstruction,TV_reconstruction,selection_operator.circular_list.mean(),selection_operator.circular_list.std(),selection_operator.max_iteration_reached_counter / optimiser.getNumberOfIndividuals(),gradient_global_fitness,gradient_TV,last_proportion_of_good_flies,1-last_proportion_of_good_flies));

        g_log_event="";


args = None;

try:

    args = checkCommandLineArguments()

    # Create test problem
    number_of_angles = args.angles[0];
    peak_value = -1;
    if not isinstance(args.peak, NoneType):
        peak_value = args.peak[0];
    k = args.initial_lambda[0];
    global_fitness_function = TomographyGlobalFitness(args.input[0],
                                                      args.objective[0],
                                                      2,
                                                      number_of_angles,
                                                      peak_value,
                                                      k);

    if not isinstance(args.save_input_images, NoneType):
        global_fitness_function.saveInputImages(args.save_input_images[0]);

    local_fitness_function = TomographyLocalFitness(global_fitness_function);


    # Parameters for EA
    number_of_individuals            = int(round( global_fitness_function.image.sum() / (256 * 2)));
    number_of_generation             = args.generations[0];

    if not isinstance(args.initial_pop_size, NoneType):
        number_of_individuals = args.initial_pop_size[0];

    # Log messages
    if not isinstance(args.logging, NoneType):
        logging.debug("Number of angles: %i", number_of_angles)
        logging.debug("Peak value for the Poisson noise: %f", peak_value)
        logging.debug("Number of individuals: %i", number_of_individuals)
        logging.debug("Number of generations: %i", number_of_generation)


    # Create the optimiser
    optimiser = EvolutionaryAlgorithm(local_fitness_function,
        number_of_individuals, global_fitness_function);

    # Default tournament size
    tournament_size = 2;

    # The tournament size is always two for dual
    if args.selection[0] == "dual":
        tournament_size = 2;
    # Update the tournament size if needed
    elif not isinstance(args.tournament_size, NoneType):
        if isinstance(args.tournament_size, int):
            tournament_size = args.tournament_size;
        else:
            tournament_size = args.tournament_size[0];


    # Set the selection operator
    tournament_selection = TournamentSelection(tournament_size);

    #optimiser.setSelectionOperator(RouletteWheelSelection());
    #optimiser.setSelectionOperator(RankSelection());

    # Set the selection operator
    selection_operator = None;
    if args.selection[0] == "dual" or args.selection[0] == "tournament":
        selection_operator = tournament_selection;
    elif args.selection[0] == "threshold":
        selection_operator = ThresholdSelection(
        0,
        tournament_selection,
        10);
    else:
        raise ValueError('Invalid selection operator "%s". Choose "threshold", "tournament" or "dual".' % (args.selection[0]))

    optimiser.setSelectionOperator(selection_operator);

    # Create the genetic operators
    new_blood = NewBloodOperator(args.initial_new_blood_probability[0]);
    gaussian_mutation = GaussianMutationOperator(1.0 - args.initial_new_blood_probability[0], args.initial_mutation_variance[0]);

    # Add the genetic operators to the EA
    optimiser.addGeneticOperator(new_blood);
    optimiser.addGeneticOperator(gaussian_mutation);


    # Show the visualisation
    if args.visualisation:
        fig, ax = plt.subplots(7,2);
        global_fitness_function.plot(fig, ax, 0, number_of_generation)

    # Create a progress bar
    bar = MyBar('Generation', max=number_of_generation)
    best_global_fitness = global_fitness_function.global_fitness_set[-1];

    # Log message
    if not isinstance(args.logging, NoneType):
        logging.debug("Initial Global fitness: %f" % best_global_fitness);
        logging.debug("Initial RMSE: %f" % global_fitness_function.global_error_term_set[-1]);
        logging.debug("Initial TV: %f" % global_fitness_function.global_regularisation_term_set[-1]);

    # Counters
    i = 0;
    stagnation = 0;
    number_of_mitosis = 0;
    g_generation = 0;

    # Run the evolutionary loop
    run_evolutionary_loop = True;

    # Log the statistics
    last_proportion_of_good_flies = 0;
    g_log_event="Random initial population"; logStatistics(optimiser.getNumberOfIndividuals()); g_generation += 1;

    if args.selection[0] == "threshold":
        selection_operator.max_iteration_reached_counter = -100;


    addToFitnessCircularList();

    previous_gradient_global_fitness = None;
    previous_gradient_TV             = None;
    previous_proportion_of_good_flies = 0;

    while run_evolutionary_loop:

        # The max number of generations has not been reached
        if i < number_of_generation:

            current_gradient_global_fitness = getGradient(x_set, global_fitness_set);
            current_gradient_TV             = getGradient(x_set, TV_set);


            # Stagnation has been reached
            stagnation_reached = False;

                # The threshold selection operator tends to get stuck
            if args.selection[0] == "threshold":
                '''print()
                print()
                print()
                print()
                print()
                print(
                    i,
                    global_fitness_function.global_fitness_set[-1],
                    current_gradient_global_fitness,
                    current_gradient_TV,
                    selection_operator.circular_list.mean(),
                    selection_operator.max_iteration_reached_counter / optimiser.getNumberOfIndividuals(),
                    optimiser.getNumberOfIndividuals())'''

                '''if not isinstance(previous_gradient_global_fitness, NoneType):
                    print(previous_gradient_global_fitness - current_gradient_global_fitness,               previous_gradient_TV - current_gradient_TV
                    );

                #if selection_operator.max_iteration_reached_counter > 0:
                if selection_operator.max_iteration_reached_counter / optimiser.getNumberOfIndividuals() > -0.65:
                    stagnation_reached = True;
                    print("Threshold selection can't find flies to kill (selection_operator.max_iteration_reached_counter / optimiser.getNumberOfIndividuals() = %f > -0.65)." % (selection_operator.max_iteration_reached_counter / optimiser.getNumberOfIndividuals()));

                    logging.debug("Threshold selection can't find flies to kill (selection_operator.max_iteration_reached_counter / optimiser.getNumberOfIndividuals() = %f > -0.65)." % (selection_operator.max_iteration_reached_counter / optimiser.getNumberOfIndividuals()));

                if selection_operator.circular_list.mean() > -0.1:
                    stagnation_reached = True;
                    print("Threshold selection can't find flies to kill (selection_operator.circular_list.mean() = %f > -0.1)." % selection_operator.circular_list.mean());

                    logging.debug("Threshold selection can't find flies to kill (selection_operator.circular_list.mean() = %f > -0.1)." % selection_operator.circular_list.mean());

                if gradient_global_fitness > 0.0:
                    stagnation_reached = True;
                    print("Gradient global fitness is no longer negative (%f)." % gradient_global_fitness);

                    logging.debug("Gradient global fitness is no longer negative (%f)." % gradient_global_fitness);
                    '''

                if selection_operator.number_of_good_flies + selection_operator.number_of_bad_flies > 0:

                    current_proportion_of_good_flies = selection_operator.number_of_good_flies / (selection_operator.number_of_good_flies + selection_operator.number_of_bad_flies);

                    last_proportion_of_good_flies = current_proportion_of_good_flies;

                    '''print("Good: ", 100*current_proportion_of_good_flies, " bad: ", 100*(1-current_proportion_of_good_flies), " old was: ", 100 * previous_proportion_of_good_flies);'''

                    if current_proportion_of_good_flies < previous_proportion_of_good_flies:
                        stagnation_reached = True;
                        '''print("There are less good flies this time  (%f) than previously (%f)." % (100 * current_proportion_of_good_flies, 100 * previous_proportion_of_good_flies));'''

                        logging.debug("There are less good flies this time  (%f) than previously (%f)." % (100 * current_proportion_of_good_flies, 100 * previous_proportion_of_good_flies));
                        previous_proportion_of_good_flies = 0;
                    else:
                        previous_proportion_of_good_flies = current_proportion_of_good_flies;

            if stagnation >= args.max_stagnation_counter[0]:
                stagnation_reached = True;
                '''print("Population stagnation (no improvement of the global fitness over %i generations)." % args.max_stagnation_counter[0]);'''
                logging.debug("Population stagnation (no improvement of the global fitness over %i generations)." % args.max_stagnation_counter[0]);

            # Reset the counter
            if args.selection[0] == "threshold":
                selection_operator.max_iteration_reached_counter = 0;
                #selection_operator.circular_list = np.ones(10) * -1;


                selection_operator.number_of_good_flies = 0;
                selection_operator.number_of_bad_flies = 0;

            if stagnation_reached:

                # Check the population size
                current_population_size = optimiser.getNumberOfIndividuals();
                target_population_size = 2 * current_population_size;

                run_mitosis = False;

                # There is no max population size
                if isinstance(args.max_pop_size, NoneType):
                    run_mitosis = True;

                # There is a max population size
                else:
                    # Has not reached the max population size
                    if target_population_size <= args.max_pop_size[0]:
                        run_mitosis = True;

                    # Has reached the max population size
                    else:
                        # Exit the for loop
                        run_evolutionary_loop = False;

                        # Log message
                        if not isinstance(args.logging, NoneType):
                            logging.debug("Stopping criteria met. Population stagnation and max population size reached. The current population size is %i, the double is %i, which is higher than the threshold %i" % (current_population_size, target_population_size, args.max_pop_size[0]));
                            '''print("Stopping criteria met. Population stagnation and max population size reached. The current population size is %i, the double is %i, which is higher than the threshold %i" % (current_population_size, target_population_size, args.max_pop_size[0]))'''

                # Perform the mitosis
                if run_mitosis:

                    # Log message
                    if not isinstance(args.logging, NoneType):
                        logging.debug("Mitosis from %i individuals to %i" % (current_population_size, target_population_size));

                    # Decrease the mutation variance
                    old_mutation_variance = gaussian_mutation.mutation_variance;
                    gaussian_mutation.mutation_variance /= 2;

                    # Perform the mitosis and log the statistics
                    optimiser.mitosis(gaussian_mutation, args.generational);
                    g_log_event="Mitosis"; logStatistics(optimiser.getNumberOfIndividuals()); g_generation += 1;

                    # Restore the mutation variance
                    gaussian_mutation.mutation_variance = old_mutation_variance;

                    # Reset the stagnation counter and
                    # Update the best global fitness
                    stagnation = 0;
                    best_global_fitness = global_fitness_function.global_fitness_set[-1];


                    # Increase the mitosis counter
                    number_of_mitosis += 1;

            # Update the operators' probability
            # Decrease the new blood operator's probability
            # Increase the mutation operator's probability
            start = args.initial_new_blood_probability[0];
            end   = args.final_new_blood_probability[0];
            new_blood.probability = linearInterpolation(start, end, i, number_of_generation - 1);
            gaussian_mutation.probability = 1.0 - new_blood.probability;

            # Decrease the mutation variance
            start = args.initial_mutation_variance[0];
            end   = args.final_mutation_variance[0];
            gaussian_mutation.mutation_variance = linearInterpolation(start, end, i, number_of_generation - 1);

            # Increase the regularisation weight
            start = args.initial_lambda[0];
            end   = args.final_lambda[0];
            global_fitness_function.k = linearInterpolation(start, end, i, number_of_generation - 1);

            # Do not update the local fitness of all the individuals
            # in steady-state EA before running the evolutionary loop
            if args.steady_state:
                optimiser.evaluateGlobalFitness(False);
                optimiser.runSteadyState();

            # Update the local fitness of all the individuals in generational EA
            # before running the evolutionary loop
            elif args.generational:
                optimiser.evaluateGlobalFitness(True);
                optimiser.runIteration();

            # Log the statistics
            g_log_event="Evolutionary loop"; logStatistics(optimiser.getNumberOfIndividuals()); g_generation += 1;

            # Get the current global fitness
            new_global_fitness = global_fitness_function.global_fitness_set[-1];

            addToFitnessCircularList();

            previous_gradient_global_fitness = current_gradient_global_fitness;
            previous_gradient_TV             = current_gradient_TV;

            # The population has not improved since the last check
            if new_global_fitness >= best_global_fitness:
                stagnation += 1; # Increase the stagnation counter

            # The population has improved since the last check
            else:
                # Reset the stagnation counter and
                # Update the best global fitness
                stagnation = 0;
                best_global_fitness = new_global_fitness;

            # Log message
            if not isinstance(args.logging, NoneType):
                logging.debug("Global fitness after %i-th generation: %f" % (i, global_fitness_function.global_fitness_set[-1]));
                logging.debug("RMSE after %i-th generation: %f" % (i, global_fitness_function.global_error_term_set[-1]));
                logging.debug("TV after %i-th generation: %f" % (i, global_fitness_function.global_regularisation_term_set[-1]));

            # Update progress bar
            bar.next();

            # Show the visualisation
            if args.visualisation:

                # The main windows is still open
                # (does not work with Tkinker backend)
                if plt.fignum_exists(fig.number) and plt.get_fignums():

                    # Update the main window
                    global_fitness_function.plot(fig, ax, i, number_of_generation)
                    plt.pause(5.00)
                    #plt.savefig('test.eps', format='eps', bbox_inches='tight', pad_inches=1.0, dpi=600)

            # Increment the counter
            i += 1;

        # The max number of generations has been reached
        else:

            # Stop the evolutionary loop
            run_evolutionary_loop = False;

            # Log messages
            if not isinstance(args.logging, NoneType):

                logging.debug("Stopping criteria met. Number of new generations (%i) reached" % number_of_generation);

    bar.finish();


    # Log messages
    if not isinstance(args.logging, NoneType):

        logging.debug("Number of global fitness evaluation: %i", global_fitness_function.number_of_calls - local_fitness_function.number_of_calls)
        logging.debug("Number of local fitness evaluation: %i", local_fitness_function.number_of_calls)


    # Show the visualisation
    if args.visualisation:

        # Create a new figure and show the reconstruction with the bad flies
        fig = plt.figure();
        fig.canvas.set_window_title("Reconstruction with all the flies (bad and good)")
        plt.imshow(global_fitness_function.population_image_data, cmap=plt.cm.Greys_r);


    # There is an output for the image with the bad flies
    if not isinstance(args.output_with_bad_flies, NoneType):

        # Save a PNG file
        imsave(args.output_with_bad_flies[0] + '-reconstruction.png', global_fitness_function.population_image_data);

        # Save an ASCII file
        np.savetxt(args.output_with_bad_flies[0] + '-reconstruction.txt', global_fitness_function.population_image_data);

        # Save a PNG file
        imsave(args.output_with_bad_flies[0] + '-projections.png', global_fitness_function.population_sinogram_data);

        # Save an ASCII file
        np.savetxt(args.output_with_bad_flies[0] + '-projections.txt', global_fitness_function.population_sinogram_data);

        # Log message
        if not isinstance(args.logging, NoneType):

            logging.debug("Global fitness before cleaning: %f", global_fitness_function.global_fitness_set[-1]);


    # Remove the bad flies
    number_of_good_individuals, number_of_bad_individuals, good_individual_set = filterBadIndividualsOut();


    # Update the global fitness
    global_fitness_function.objectiveFunction(good_individual_set, True);

    # Log the statistics
    g_log_event = "remove bad flies"; logStatistics(number_of_good_individuals); g_generation += 1;

    # Log messages
    if not isinstance(args.logging, NoneType):

        logging.debug("Total number of good individuals: %i", number_of_good_individuals);
        logging.debug("Total number of bad individuals: %i", len(optimiser.current_solution_set) - number_of_good_individuals);
        logging.debug("Global fitness after cleaning: %f", global_fitness_function.global_fitness_set[-1]);


    # Show the visualisation
    if args.visualisation:

        # The main windows is still open (does not work with Tkinker backend)
        if plt.fignum_exists(fig.number) and plt.get_fignums():

            # Update the main window
            global_fitness_function.plot(fig, ax, i, number_of_generation);
            i += 1;
            plt.pause(5.00)
            #plt.savefig('test.eps', format='eps', bbox_inches='tight', pad_inches=1.0, dpi=600)

        # Create a new figure and show the reconstruction without the bad flies
        fig = plt.figure();
        fig.canvas.set_window_title("Reconstruction without the bad flies")
        plt.imshow(global_fitness_function.population_image_data, cmap=plt.cm.Greys_r);


    # There is an output for the image without the bad flies
    if not isinstance(args.output_without_bad_flies, NoneType):

        # Save a PNG file
        imsave(args.output_without_bad_flies[0] + '-reconstruction.png', global_fitness_function.population_image_data);

        # Save an ASCII file
        np.savetxt(args.output_without_bad_flies[0] + '-reconstruction.txt', global_fitness_function.population_image_data);

        # Save a PNG file
        imsave(args.output_without_bad_flies[0] + '-projections.png', global_fitness_function.population_sinogram_data);

        # Save an ASCII file
        np.savetxt(args.output_without_bad_flies[0] + '-projections.txt', global_fitness_function.population_sinogram_data);

    # Show the visualisation
    if args.visualisation:

        # The main windows is still open (does not work with Tkinker backend)
        if plt.fignum_exists(fig.number) and plt.get_fignums():

            # Update the main window
            global_fitness_function.plot(fig, ax, i, number_of_generation);
            i += 1;

        # Show all the windows
        plt.show();

except Exception as e:
    if not isinstance(args.logging, NoneType):
        logging.critical("Exception occurred", exc_info=True)
    else:
        print(e)
    sys.exit(os.EX_SOFTWARE)

sys.exit(os.EX_OK) # code 0, all ok
