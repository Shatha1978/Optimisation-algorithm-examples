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

import ImageMetrics as IM;

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

    parser.add_argument('--output', help='Reconstructed image',      nargs=1, type=str, required=False);

    parser.add_argument('--save_input_images', help='Where to save the input images (groundtruth with and without noise, and the sinogram)',      nargs=1, type=str, required=False);

    parser.add_argument('--angles', help='Number of angles',      nargs=1, type=int, required=True);

    parser.add_argument('--peak', help='Peak value for the Poisson noise',      nargs=1, type=float, required=False);

    parser.add_argument('--selection', help='Selection operator (ranking, roulette, tournament or dual)',      nargs=1, type=str, required=True);

    parser.add_argument('--pop_size', help='Size of the population',      nargs=1, type=int, required=True);

    parser.add_argument('--number_of_emission_points', help='Number of emission points',      nargs=1, type=int, required=False);

    parser.add_argument('--tournament_size', help='Number of individuals involved in the tournament',      nargs=1, type=int, required=False, default=2);

    parser.add_argument('--generations', help='Number of generations',      nargs=1, type=int, required=True);

    parser.add_argument('--visualisation', help='Realtime visualisation', action="store_true");

    parser.add_argument('--max_stagnation_counter', help='Max value of the stagnation counter to trigger a mitosis', nargs=1, type=int, required=True);

    parser.add_argument('--initial_lambda', help='Weight of the TV-norm regularisation at the start of the optimisation', nargs=1, type=float, required=True);

    parser.add_argument('--final_lambda', help='Weight of the TV-norm regularisation at the end of the optimisation', nargs=1, type=float, required=True);

    parser.add_argument('--initial_mutation_variance', help='Mutation variance at the start of the optimisation', nargs=1, type=float, required=True);

    parser.add_argument('--final_mutation_variance', help='Mutation variance at the end of the optimisation', nargs=1, type=float, required=True);

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

    if not isinstance(args.objective, NoneType):

        if args.objective[0] not in IM.MINIMISATION and args.objective[0] not in IM.MAXIMISATION:
            raise ValueError('Argument --objective "%s" is not valid.' % args.objective[0])

    return args;


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

def logStatistics(aNumberOfIndividuals):

    global global_fitness_function;
    global g_first_log;
    global g_log_event;
    global g_generation;
    global optimiser;

    if not isinstance(args.logging, NoneType):
        if g_first_log:
            g_first_log = False;
            logging.info("generation,new_individual_counter,event,number_of_emission_points,MAE_sinogram,MSE_sinogram,RMSE_sinogram,NRMSE_euclidean_sinogram,NRMSE_mean_sinogram,NRMSE_min_max_sinogram,cosine_similarity_sinogram,SSIM_sinogram,PSNR_sinogram,ZNCC_sinogram,TV_sinogram,MAE_reconstruction,MSE_reconstruction,RMSE_reconstruction,NRMSE_euclidean_reconstruction,NRMSE_mean_reconstruction,NRMSE_min_max_reconstruction,cosine_similarity_reconstruction,SSIM_reconstruction,PSNR_reconstruction,ZNCC_reconstruction,TV_reconstruction");

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

        logging.info("%i,%i,%s,%i,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f" % (g_generation,optimiser.number_created_children,g_log_event,aNumberOfIndividuals,MAE_sinogram,MSE_sinogram,RMSE_sinogram,NRMSE_euclidean_sinogram,NRMSE_mean_sinogram,NRMSE_min_max_sinogram,cosine_similarity_sinogram,SSIM_sinogram,PSNR_sinogram,ZNCC_sinogram,TV_sinogram,MAE_reconstruction,MSE_reconstruction,RMSE_reconstruction,NRMSE_euclidean_reconstruction,NRMSE_mean_reconstruction,NRMSE_min_max_reconstruction,cosine_similarity_reconstruction,SSIM_reconstruction,PSNR_reconstruction,ZNCC_reconstruction,TV_reconstruction));

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
                                                      args.number_of_emission_points[0],
                                                      number_of_angles,
                                                      peak_value,
                                                      k);
    global_fitness_function.save_best_solution = True;

    if not isinstance(args.save_input_images, NoneType):
        global_fitness_function.saveInputImages(args.save_input_images[0]);

    # Parameters for EA
    number_of_individuals = args.pop_size[0];
    number_of_generation  = args.generations[0];

    # Log messages
    if not isinstance(args.logging, NoneType):
        logging.debug("Number of angles: %i", number_of_angles)
        logging.debug("Peak value for the Poisson noise: %f", peak_value)
        logging.debug("Number of individuals: %i", number_of_individuals)
        logging.debug("Number of generations: %i", number_of_generation)


    # Create the optimiser
    optimiser = EvolutionaryAlgorithm(global_fitness_function,
        number_of_individuals);

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
    if args.selection[0] == "dual" or args.selection[0] == "tournament":
        optimiser.setSelectionOperator(TournamentSelection(tournament_size));
    elif args.selection[0] == "ranking":
        optimiser.setSelectionOperator(RankSelection());
    elif args.selection[0] == "roulette":
        optimiser.setSelectionOperator(RouletteWheelSelection());
    else:
        raise ValueError('Invalid selection operator "%s". Choose "threshold", "tournament" or "dual".' % (args.selection[0]))

    # Create the genetic operators
    gaussian_mutation = GaussianMutationOperator(0.3, args.initial_mutation_variance[0]);
    blend_cross_over = BlendCrossoverOperator(0.6, gaussian_mutation);

    # Add the genetic operators to the EA
    optimiser.addGeneticOperator(blend_cross_over);
    optimiser.addGeneticOperator(gaussian_mutation);
    optimiser.addGeneticOperator(ElitismOperator(0.1));


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
    g_log_event="Random initial population"; logStatistics(optimiser.getNumberOfIndividuals()); g_generation += 1;

    while run_evolutionary_loop:

        # The max number of generations has not been reached
        if i < number_of_generation:

            # Stagnation has been reached
            if stagnation >= args.max_stagnation_counter[0]:

                # Exit the for loop
                run_evolutionary_loop = False;

                # Log message
                if not isinstance(args.logging, NoneType):
                    logging.debug("Stopping criteria met. Population stagnation.");

            # Decrease the mutation variance
            start = args.initial_mutation_variance[0];
            end   = args.final_mutation_variance[0];
            gaussian_mutation.mutation_variance = linearInterpolation(start, end, i, number_of_generation - 1);

            # Increase the regularisation weight
            start = args.initial_lambda[0];
            end   = args.final_lambda[0];
            global_fitness_function.k = linearInterpolation(start, end, i, number_of_generation - 1);

            # Run the evolutionary loop
            optimiser.runIteration();

            # Log the statistics
            g_log_event="Evolutionary loop"; logStatistics(optimiser.getNumberOfIndividuals()); g_generation += 1;

            # Get the current global fitness
            new_global_fitness = global_fitness_function.global_fitness_set[-1];

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


    # Show the visualisation
    if args.visualisation:

        # Create a new figure and show the reconstruction with the bad flies
        fig = plt.figure();
        fig.canvas.set_window_title("Reconstruction")
        plt.imshow(global_fitness_function.population_image_data, cmap=plt.cm.Greys_r);

        # Show all the windows
        plt.show();

    # There is an output for the image with the bad flies
    if not isinstance(args.output, NoneType):

        # Save a PNG file
        imsave(args.output[0] + '-reconstruction.png', global_fitness_function.population_image_data);

        # Save an ASCII file
        np.savetxt(args.output[0] + '-reconstruction.txt', global_fitness_function.population_image_data);

        # Save a PNG file
        imsave(args.output[0] + '-projections.png', global_fitness_function.population_sinogram_data);

        # Save an ASCII file
        np.savetxt(args.output[0] + '-projections.txt', global_fitness_function.population_sinogram_data);

        # Log message
        if not isinstance(args.logging, NoneType):

            logging.debug("Best global fitness: %f", global_fitness_function.global_fitness_set[-1]);


except Exception as e:
    if not isinstance(args.logging, NoneType):
        logging.critical("Exception occurred", exc_info=True)
    else:
        print(e)
    sys.exit(os.EX_SOFTWARE)

sys.exit(os.EX_OK) # code 0, all ok
