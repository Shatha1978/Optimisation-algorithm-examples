# Import the numpy package to compute the image and objective function
import numpy as np

# Import the superclass (also called base class), which is an abstract class,
# to implement the subclass AckleyFunction
from ObjectiveFunction import *

from ImageMetrics import *


# The subclass that inherits of ObjectiveFunction
class LampGlobalFitnessFunction(ObjectiveFunction):

    # Constructor
    # aNumberOfDimensions: the number of dimensions (e.g. how many parameters)
    def __init__(self, aWidth, aLength, aNumberOfLamps, aLampRadius):

        # Store the class attributes
        self.room_width  = aWidth;  # cols of the image
        self.room_length = aLength; # rows of the image
        self.lamp_radius = aLampRadius;
        self.number_of_lamps = aNumberOfLamps;

        # Store the boundaries
        self.boundaries = [];
        for _ in range(self.number_of_lamps):
            self.boundaries.append([0, self.room_width  - 1]);
            self.boundaries.append([0, self.room_length - 1]);
            self.boundaries.append([0, 1]);

        # Call the constructor of the superclass
        super().__init__(3 * aNumberOfLamps,
                         self.boundaries,
                         self.objectiveFunction,
                         1);

        # The name of the function
        self.name = "Lamp Problem";
        self.reference_image = np.ones((self.room_length, self.room_width));


    # objectiveFunction implements the Ackley function
    def objectiveFunction(self, aSolution):
        return getRMSE(self.reference_image, self.createImage(aSolution));

    def createImage(self, aSolution):
        # Create a black image
        test_image = np.zeros((self.room_length, self.room_width));

        # Process every lamp
        for i in range(round(len(aSolution) / 3)):
            # The lamp is on
            if aSolution[i * 3 + 2] > 0.5:
                # Add the lamp
                test_image = np.add(test_image, create_circular_mask(self.room_width, self.room_length, aSolution[i * 3], aSolution[i * 3 + 1], self.lamp_radius));

        # Return the new image
        return test_image;

    def saveImage(self, aSolution, aFileName):
        test_image = self.createImage(aSolution.parameter_set);
        np.savetxt(aFileName, test_image);

def create_circular_mask(w, h, x, y, radius):

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - x)**2 + (Y-y)**2)

    mask = dist_from_center <= radius
    return mask
