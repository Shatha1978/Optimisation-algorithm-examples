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



from PSO import *

from LampProblemGlobalFitness import LampProblemGlobalFitness

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

    parser.add_argument('--output', help='Reconstructed image',      nargs=1, type=str, required=False);

    parser.add_argument('--weight', help='The weight in the objective function',      nargs=1, type=float, required=True);

    parser.add_argument('--radius', help='The radius of lamps',      nargs=1, type=int, required=True);

    parser.add_argument('--room_width', help='The width of the room',      nargs=1, type=int, required=True);

    parser.add_argument('--room_height', help='The height of the room',      nargs=1, type=int, required=True);

    parser.add_argument('--swarm_size', help='Size of the swarm (number of particles)',      nargs=1, type=int, required=True);

    parser.add_argument('--number_of_lamps', help='Number of lamps',      nargs=1, type=int, required=False);

    parser.add_argument('--iterations', help='Number of iterations',      nargs=1, type=int, required=True);

    parser.add_argument('--visualisation', help='Realtime visualisation', action="store_true");

    parser.add_argument('--max_stagnation_counter', help='Max value of the stagnation counter to trigger a mitosis', nargs=1, type=int, required=True);

    parser.add_argument('--logging', help='File name of the log file', nargs=1, type=str, required=False);

    args = parser.parse_args();

    # Set the logger if needed
    if not isinstance(args.logging, NoneType):
        logging.basicConfig(filename=args.logging[0],
                            level=logging.DEBUG,
                            filemode='w',
                            format='%(asctime)s, %(name)s - %(levelname)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')

        logging.debug(args)

    return args;


class MyBar(IncrementalBar):
    suffix = '%(index)d/%(max)d - %(percent).1f%% - %(eta)ds - Best fitness %(global_fitness).5f - Average fitness %(average_fitness).5f - RMSE %(RMSE).5f - TV %(TV).5f%%'
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
    def average_fitness(self):
        global optimiser;
        return optimiser.average_objective_value

def linearInterpolation(start, end, i, j):
    return start + (end - start) * (1 - (j - i) / j);


g_first_log = True;
g_log_event = "";
g_iteration = 0;

def logStatistics(aNumberOfIndividuals):

    global global_fitness_function;
    global g_first_log;
    global g_log_event;
    global g_iteration;
    global optimiser;

    if not isinstance(args.logging, NoneType):
        if g_first_log:
            g_first_log = False;
            logging.info("generation,new_individual_counter,event,number_of_emission_points,MAE_reconstruction,MSE_reconstruction,RMSE_reconstruction,NRMSE_euclidean_reconstruction,NRMSE_mean_reconstruction,cosine_similarity_reconstruction,SSIM_reconstruction,TV_reconstruction,Average_fitness_value");

        ref  =  global_fitness_function.ground_truth;
        test = global_fitness_function.population_image_data;
        MAE_reconstruction                 = IM.getMAE(ref, test);
        MSE_reconstruction                 = IM.getMSE(ref, test);
        RMSE_reconstruction                = IM.getRMSE(ref, test);
        NRMSE_euclidean_reconstruction     = IM.getNRMSE_euclidean(ref, test);
        NRMSE_mean_reconstruction          = IM.getNRMSE_mean(ref, test);
        #NRMSE_min_max_reconstruction       = IM.getNRMSE_minMax(ref, test);
        cosine_similarity_reconstruction   = IM.getCosineSimilarity(ref, test);
        #mean_relative_error_reconstruction = IM.getMeanRelativeError(ref, test);
        #max_relative_error_reconstruction  = IM.getMaxRelativeError(ref, test);
        SSIM_reconstruction                = IM.getSSIM(ref, test);
        #PSNR_reconstruction                = IM.getPSNR(ref, test);
        #ZNCC_reconstruction                = IM.getNCC(ref, test);
        TV_reconstruction                  = IM.getTV(test);

        #logging.info("%i,%s,%i,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f" % (g_iteration,g_log_event,MAE_sinogram,MSE_sinogram,RMSE_sinogram,NRMSE_euclidean_sinogram,NRMSE_mean_sinogram,NRMSE_min_max_sinogram,cosine_similarity_sinogram,mean_relative_error_sinogram,max_relative_error_sinogram,SSIM_sinogram,PSNR_sinogram,ZNCC_sinogram,TV_sinogram,MAE_reconstruction,MSE_reconstruction,RMSE_reconstruction,NRMSE_euclidean_reconstruction,NRMSE_mean_reconstruction,NRMSE_min_max_reconstruction,cosine_similarity_reconstruction,mean_relative_error_reconstruction,max_relative_error_reconstruction,SSIM_reconstruction,PSNR_reconstruction,ZNCC_reconstruction,TV_reconstruction));

        logging.info("%i,%i,%s,%i,%f,%f,%f,%f,%f,%f,%f,%f,%f" % (g_iteration,optimiser.number_created_particles+optimiser.number_moved_particles,g_log_event,aNumberOfIndividuals,MAE_reconstruction,MSE_reconstruction,RMSE_reconstruction,NRMSE_euclidean_reconstruction,NRMSE_mean_reconstruction,cosine_similarity_reconstruction,SSIM_reconstruction,TV_reconstruction,optimiser.average_objective_value));

        g_log_event="";


args = None;

try:

    args = checkCommandLineArguments()

    # Parameters for PSO
    number_of_particles = args.swarm_size[0];
    number_of_iterations  = args.iterations[0];

    # Log messages
    if not isinstance(args.logging, NoneType):
        logging.debug("Weight: %f",                args.weight[0])
        logging.debug("Radius: %i",                args.radius[0])
        logging.debug("Room width: %i",            args.room_width[0])
        logging.debug("Room height: %i",           args.room_height[0])
        logging.debug("Number of lamps: %i",       args.number_of_lamps[0])
        logging.debug("Number of individuals: %i", number_of_particles)
        logging.debug("Number of generations: %i", number_of_iterations)

    # Create test problem
    global_fitness_function = LampProblemGlobalFitness(args.radius[0],
            args.room_width[0],
            args.room_height[0],
            args.weight[0],
            args.number_of_lamps[0]);

    global_fitness_function.save_best_solution = True;


    # Create the optimiser
    optimiser = PSO(global_fitness_function,
        number_of_particles);

    global_fitness_function.average_fitness_set.append(optimiser.average_objective_value);
    global_fitness_function.best_fitness_set.append(global_fitness_function.global_fitness_set[-1]);

    # Show the visualisation
    if args.visualisation:
        fig, ax = plt.subplots(2,2);
        global_fitness_function.plot(fig, ax, 0, number_of_iterations)

    # Create a progress bar
    bar = MyBar('Generation', max=number_of_iterations)
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
    g_iteration = 0;

    # Run the optimisation loop
    run_optimisation_loop = True;

    # Log the statistics
    g_log_event="Random initial swarm"; logStatistics(optimiser.getNumberOfParticles()); g_iteration += 1;

    while run_optimisation_loop:

        # The max number of generations has not been reached
        if i < number_of_iterations:

            # Stagnation has been reached
            if stagnation >= args.max_stagnation_counter[0] and args.max_stagnation_counter[0] > 0:

                # Exit the for loop
                run_optimisation_loop = False;

                # Log message
                if not isinstance(args.logging, NoneType):
                    logging.debug("Stopping criteria met. Population stagnation.");

            # Run the evolutionary loop
            optimiser.runIteration();

            global_fitness_function.average_fitness_set.append(optimiser.average_objective_value);
            global_fitness_function.best_fitness_set.append(global_fitness_function.global_fitness_set[-1]);

            # Log the statistics
            g_log_event="Optimisation loop"; logStatistics(optimiser.getNumberOfParticles()); g_iteration += 1;

            # Get the current global fitness
            new_global_fitness = global_fitness_function.global_fitness_set[-1];

            # The population has not improved since the last check
            if new_global_fitness >= best_global_fitness:
                stagnation += 1; # Increase the stagnation counter

            # The swarm has improved since the last check
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
                    global_fitness_function.plot(fig, ax, i, number_of_iterations)
                    plt.pause(5.00)
                    #plt.savefig('test.eps', format='eps', bbox_inches='tight', pad_inches=1.0, dpi=600)

            # Increment the counter
            i += 1;

        # The max number of generations has been reached
        else:

            # Stop the evolutionary loop
            run_optimisation_loop = False;

            # Log messages
            if not isinstance(args.logging, NoneType):

                logging.debug("Stopping criteria met. Number of new generations (%i) reached" % number_of_iterations);

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
