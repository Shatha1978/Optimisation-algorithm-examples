import math

import numpy as np

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


def normalise(image):
    return (image - image.mean()) / image.std();

class TomographyGlobalFitness(ObjectiveFunction):
    def __init__(self, anInputImage, anObjective, aSearchSpaceDimension = 2, aNumberOfAngles=180, aPeakValue = 100, k = -1):

        self.loadImageData(anInputImage, aNumberOfAngles, aPeakValue);

        # Store the image simulated by the flies
        self.population_image_data = np.zeros(self.noisy.shape, self.noisy.dtype)

        self.population_sinogram_data = np.zeros(self.projections.shape, self.projections.dtype)

        self.fig = None;
        ax  = None;
        self.global_fitness_set = [];
        self.global_error_term_set = [];
        self.global_regularisation_term_set = [];
        self.zncc_set = [];
        self.k = k;
        self.current_population = None;
        self.number_of_calls = 0;
        self.save_best_solution = False;

        if anObjective == "SAE":
            type_of_optimisation = ObjectiveFunction.MINIMISATION
            self.image_metrics_function = IM.getSAE;
        elif anObjective == "SSE":
            type_of_optimisation = ObjectiveFunction.MINIMISATION
            self.image_metrics_function = IM.getSSE;
        elif anObjective == "MAE":
            type_of_optimisation = ObjectiveFunction.MINIMISATION
            self.image_metrics_function = IM.getMAE;
        elif anObjective == "MSE":
            type_of_optimisation = ObjectiveFunction.MINIMISATION
            self.image_metrics_function = IM.getMSE;
        elif anObjective == "RMSE":
            type_of_optimisation = ObjectiveFunction.MINIMISATION
            self.image_metrics_function = IM.getRMSE;
        elif anObjective == "NRMSE_euclidean":
            type_of_optimisation = ObjectiveFunction.MINIMISATION
            self.image_metrics_function = IM.getNRMSE_euclidean;
        elif anObjective == "NRMSE_mean":
            type_of_optimisation = ObjectiveFunction.MINIMISATION
            self.image_metrics_function = IM.getNRMSE_mean;
        elif anObjective == "NRMSE_minMax":
            type_of_optimisation = ObjectiveFunction.MINIMISATION
            self.image_metrics_function = IM.getNRMSE_minMax;
        elif anObjective == "mean_relative_error":
            type_of_optimisation = ObjectiveFunction.MINIMISATION
            self.image_metrics_function = IM.getMeanRelativeError;
        elif anObjective == "max_relative_error":
            type_of_optimisation = ObjectiveFunction.MINIMISATION
            self.image_metrics_function = IM.getMaxRelativeError;
        elif anObjective == "cosine_similarity":
            type_of_optimisation = ObjectiveFunction.MAXIMISATION
            self.image_metrics_function = IM.getCosineSimilarity;
        elif anObjective == "SSIM":
            type_of_optimisation = ObjectiveFunction.MAXIMISATION
            self.image_metrics_function = IM.getSSIM;
        elif anObjective == "PSNR":
            type_of_optimisation = ObjectiveFunction.MAXIMISATION
            self.image_metrics_function = IM.getPSNR;
        elif anObjective == "NCC" or anObjective == "ZNCC":
            type_of_optimisation = ObjectiveFunction.MAXIMISATION
            self.image_metrics_function = IM.getNCC;
        else:
            raise ValueError('Invalid objective function "%s".' % (anObjective));

        self.boundaries = [];
        for _ in range(aSearchSpaceDimension):
            self.boundaries.append([0, max(self.noisy.shape) - 1]);
            self.boundaries.append([0, max(self.noisy.shape) - 1]);

        super().__init__(2 * aSearchSpaceDimension,
                         self.boundaries,
                         self.objectiveFunction,
                         type_of_optimisation);

        self.name = "anObjective";

    def objectiveFunction(self, aParameterSet, aSavePopulationFlag = True):

        self.number_of_calls += 1;

        image_data = np.zeros(self.noisy.shape, self.noisy.dtype)

        individual_weight = self.total_weight / (len(aParameterSet) / 2);

        for i,j in zip(aParameterSet[0::2], aParameterSet[1::2]):
            x = math.floor(i);
            y = math.floor(j);

            image_data[y,x] += individual_weight;

        sinogram_data = radon(image_data, theta=self.theta, circle=False)


        error_term = self.image_metrics_function(self.projections, sinogram_data);
        fitness = error_term;

        tv_norm = 0.5 * IM.getTV(image_data);

        if self.k > 0.0:

            regularisation_term = self.k * tv_norm;
            fitness += regularisation_term;

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
                self.population_sinogram_data = sinogram_data;
                self.global_fitness_set.append(fitness);
                self.global_error_term_set.append(error_term);
                self.global_regularisation_term_set.append(tv_norm);
                self.zncc_set.append(IM.getNCC(self.image, self.population_image_data));

        return fitness;


    def loadImageData(self, anInputImage, aNumberOfAngles, aPeakValue):
        # Load the phantom (considered as unknown)
        data_dir = '.';

        if os.path.splitext(anInputImage)[1] == ".txt":
            image = np.loadtxt(anInputImage);
        else:
            image = imread(anInputImage, as_gray=True)

        # Zoom out
        image = zoom(image, 0.5)

        # Convert from uint8 to float
        self.image = image.astype(np.float)

        # Add some noise using the Poisson distribution
        if aPeakValue > 0.0:
            self.noisy = np.random.poisson(image / 255.0 * aPeakValue) /   aPeakValue * 255  # noisy image
        # Do not add noise
        else:
            self.noisy = self.image;

        # Compute the Radon transform
        self.theta = np.linspace(0., 180., aNumberOfAngles, endpoint=False)
        self.projections = radon(self.noisy, theta=self.theta, circle=False)

        self.total_weight = np.sum(self.noisy);

        # Perform the FBP reconstruction
        self.fbp_reconstruction = iradon(self.projections,
            theta=self.theta,
            filter="hann",
            interpolation="cubic",
            circle=False)

        self.FBP_zncc = IM.getNCC(self.image, self.fbp_reconstruction);

        # Perform the SART reconstruction
        self.sart_reconstruction = IM.cropCenter(iradon_sart(self.projections,
            theta=self.theta, relaxation=0.05), self.image.shape[1], self.image.shape[0]);

        self.SART_zncc = IM.getNCC(self.image, self.sart_reconstruction);

    def saveInputImages(self, aFilePrefix = ""):
        prefix = aFilePrefix;

        # Groundtruth

        # Save a PNG file
        imsave(prefix + '-groundtruth.png', self.image);

        # Save an ASCII file
        np.savetxt(prefix + '-groundtruth.txt', self.image);

        # Noisy

        # Save a PNG file
        imsave(prefix + '-noisy.png', self.noisy);

        # Save an ASCII file
        np.savetxt(prefix + '-noisy.txt', self.noisy);

        # Sinogram

        # Save a PNG file
        imsave(prefix + '-sinogram.png', self.projections);

        # Save an ASCII file
        np.savetxt(prefix + '-sinogram.txt', self.projections);



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
            ax[0, 0].imshow(self.image, cmap=plt.cm.Greys_r)

            # Plot the noisy image
            ax[0, 1].set_title("Noisy");
            ax[0, 1].imshow(self.noisy, cmap=plt.cm.Greys_r)

            # Plot some projections
            projections = radon(self.noisy, theta=theta, circle=False)
            title = "Projections at\n";

            for i in range(len(theta) - 1):
                title += str(theta[i]) + ", ";

            title += 'and ' + \
                str(theta[len(theta) - 1]) + \
                " degrees";

            ax[1, 0].plot(projections);
            ax[1, 0].set_title(title)
            ax[1, 0].set_xlabel("Projection axis");
            ax[1, 0].set_ylabel("Intensity");

            # Plot the sinogram
            ax[1, 1].set_title("Radon transform\n(Sinogram)");
            ax[1, 1].set_xlabel("Projection axis");
            ax[1, 1].set_ylabel("Intensity");
            ax[1, 1].imshow(self.projections)

            # Plot the FBP reconstruction
            ax[2, 0].set_title("FBP reconstruction")
            ax[2, 0].imshow(self.fbp_reconstruction, cmap=plt.cm.Greys_r)

            # Plot the FBP reconstruction error map
            ax[2, 1].set_title("FBP reconstruction error")
            ax[2, 1].imshow(self.fbp_reconstruction - self.image, cmap=plt.cm.Greys_r)

            # Plot the SART reconstruction
            ax[3, 0].set_title("SART reconstruction")
            ax[3, 0].imshow(self.sart_reconstruction, cmap=plt.cm.Greys_r)

            # Plot the SART reconstruction error map
            ax[3, 1].set_title("SART reconstruction error")
            ax[3, 1].imshow(self.sart_reconstruction - self.image, cmap=plt.cm.Greys_r)

            # Plot some projections
            ax[4, 0].set_title(title)
            ax[4, 0].set_xlabel("Projection axis");
            ax[4, 0].set_ylabel("Intensity");

            # Plot the sinogram
            ax[4, 1].set_title("Radon transform\n(Sinogram)");
            ax[4, 1].set_xlabel("Projection axis");
            ax[4, 1].set_ylabel("Intensity");

            # Plot the Evolutionary reconstruction
            ax[5, 0].set_title("Evolutionary reconstruction")

            # Plot the Evolutionary reconstruction error map
            ax[5, 1].set_title("Evolutionary reconstruction error")

            # Plot the global fitness
            ax[6, 0].set_title("Global fitness")

            ax[6, 1].set_title("Reconstruction ZNCC")
            ax[6, 1].legend(loc='lower right')
            plt.subplots_adjust(hspace=0.4, wspace=0.5)

        projections = radon(self.population_image_data, theta=theta, circle=False)
        ax[4, 0].clear();
        ax[4, 0].plot(projections);

        # Plot the sinogram
        ax[4, 1].imshow(self.population_sinogram_data)

        # Plot the Evolutionary reconstruction
        ax[5, 0].imshow(self.population_image_data, cmap=plt.cm.Greys_r)

        # Plot the Evolutionary reconstruction error map
        ax[5, 1].imshow(self.population_image_data - self.image, cmap=plt.cm.Greys_r)

        ax[6, 0].clear();
        ax[6, 0].plot(self.global_fitness_set);

        ax[6, 1].clear();
        ax[6, 1].plot(np.full(len(self.zncc_set), self.FBP_zncc), label="FBP");
        ax[6, 1].plot(np.full(len(self.zncc_set), self.SART_zncc), label="SART");
        ax[6, 1].plot(self.zncc_set, label="FA");
