import math

import numpy as np
import cv2

import matplotlib.pyplot as plt

from skimage.io import imread, imsave
#from skimage import data_dir
from skimage.transform import radon, iradon, iradon_sart
from scipy.ndimage import zoom
from sklearn import preprocessing


from ObjectiveFunction import *
import ImageMetrics as IM;

import os.path # For file extension


NoneType = type(None);


class LampProblemGlobalFitness(ObjectiveFunction):
    def __init__(self, aLampRadius, aRoomWidth, aRoomHeight, W, aSearchSpaceDimension):

        self.room_width  = aRoomWidth;
        self.room_height = aRoomHeight;
        self.lamp_radius = aLampRadius;
        self.W = W;

        # Ground truth
        self.ground_truth = np.ones((self.room_height, self.room_width), np.float32)

        # Store the image simulated by the flies
        self.population_image_data = np.zeros((self.room_height, self.room_width), np.float32)

        self.fig = None;
        ax  = None;
        self.global_fitness_set = [];
        self.global_error_term_set = [];
        self.global_regularisation_term_set = [];
        self.current_population = None;
        self.number_of_calls = 0;
        self.save_best_solution = False;

        type_of_optimisation = ObjectiveFunction.MAXIMISATION;

        self.boundaries = [];
        for _ in range(aSearchSpaceDimension):
            self.boundaries.append([0, self.room_width - 1]);
            self.boundaries.append([0, self.room_height - 1]);

        super().__init__(2 * aSearchSpaceDimension,
                         self.boundaries,
                         self.objectiveFunction,
                         type_of_optimisation);

        self.name = "anObjective";

    def getArea(self):
        return self.room_width * self.room_height;


    def createLampMap(self, aParameterSet):
        image_data = np.zeros((self.room_height, self.room_width), np.float32)

        for i,j in zip(aParameterSet[0::2], aParameterSet[1::2]):
            x = math.floor(i);
            y = math.floor(j);

            self.addLampToImage(image_data, x, y, 1);

        return image_data;

    def addLampToImage(self, overlay_image, x, y, l):

        # Draw circles corresponding to the lamps
        black_image = np.zeros((self.room_height, self.room_width), np.float32)
        cv2.circle(black_image, (x,y), self.lamp_radius, (l, l, l), -1)
        np.add(overlay_image, black_image, overlay_image);

    def areaEnlightened(self, overlay_image):
        return np.array(overlay_image).sum();

    def areaOverlap(self, overlay_image):

        areaOver = 0
        for i in range(overlay_image.shape[0]):
            for j in range(overlay_image.shape[1]):

                if (overlay_image[i,j] > 1.5):
                    areaOver += overlay_image[i,j] - 1;

        return areaOver;

    def objectiveFunction(self, aParameterSet, aSavePopulationFlag = True):

        self.number_of_calls += 1;

        image_data = self.createLampMap(aParameterSet);

        area_enlightened = self.areaEnlightened(image_data);
        overlap          = self.areaOverlap(image_data);
        fitness = (area_enlightened - self.W * overlap) / self.getArea();




        error_term = IM.getRMSE(self.ground_truth, image_data);
        #fitness = error_term;

        tv_norm = 0.5 * IM.getTV(image_data);

        if aSavePopulationFlag:

            save_data = True;

            if len(self.global_fitness_set) > 0 and self.save_best_solution:
                if self.flag == ObjectiveFunction.MINIMISATION and self.global_fitness_set[-1] < fitness:
                    save_data = False;
                elif self.flag == ObjectiveFunction.MAXIMISATION and self.global_fitness_set[-1] > fitness:
                    save_data = False;

            if save_data:
                self.current_population = copy.deepcopy(aParameterSet);
                self.population_image_data = image_data;
                self.global_fitness_set.append(fitness);
                self.global_error_term_set.append(error_term);
                self.global_regularisation_term_set.append(tv_norm);

        return fitness;


    def plot(self, fig, ax, aGenerationID, aTotalNumberOfGenerations):

        window_title = "Generation " + str(aGenerationID) + "/" + str(aTotalNumberOfGenerations) + " - Global fitness: " + str(self.global_fitness_set[-1]);

        fig.canvas.set_window_title(window_title)
        theta = [];
        theta.append(self.theta[0])

        if theta[-1] != self.theta[math.floor(len(self.theta) * 0.25)]:
            theta.append(self.theta[math.floor(len(self.theta) * 0.25)])

        if theta[-1] != self.theta[math.floor(len(self.theta) * 0.5)]:
            theta.append(self.theta[math.floor(len(self.theta) * 0.5)])

        if theta[-1] != self.theta[math.floor(len(self.theta) * 0.75)]:
            theta.append(self.theta[math.floor(len(self.theta) * 0.75)])

        #plt.axis([0, 10, 0, 1])

        # Create a figure using Matplotlib
        # It constains 5 sub-figures
        if isinstance(self.fig, NoneType):

            self.fig = 1;

            # Plot the original image
            ax[0, 0].set_title("Original");
            ax[0, 0].imshow(self.ground_truth, cmap=plt.cm.Greys_r)

            # Plot the image from the flies
            ax[0, 1].set_title("Lamps");
            ax[0, 1].imshow(self.population_image_data, cmap=plt.cm.Greys_r)