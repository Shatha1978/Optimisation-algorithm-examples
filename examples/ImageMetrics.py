import math
import copy;

import numpy as np

import skimage.measure as measure;
import skimage.exposure as exposure;

def getEntropy(anImage):
    grayImg = (linearNormalisation(anImage, 0, 255)).astype(np.uint8);
    return measure.shannon_entropy(grayImg);

def zeroMeanNormalisation(anImage):
    return (anImage - anImage.mean()) / (anImage.std());

def linearNormalisation(anImage, aMinValue = 0, aMaxValue = 1):
    return aMinValue + (aMaxValue - aMinValue) * (anImage - anImage.mean()) / (anImage.std());

def normalise(anImage):

    #return zeroMeanNormalisation(anImage);
    return linearNormalisation(anImage);
    #return copy.deepcopy(anImage);

def productImage(anImage1, anImage2):
    return (np.multiply(anImage1, anImage2));

def getHistogram(anImage, aNumberOfBins):
    return exposure.histogram(anImage, aNumberOfBins);

def getMAE(aReferenceVector, aTestVector):
    return np.abs(np.subtract(aReferenceVector, aTestVector)).mean();

def getCosineSimilarity(aReferenceVector, aTestVector):
    return np.dot(aReferenceVector, aTestVector) / (LA.norm(aReferenceVector) * LA.norm(aTestVector))

def getMeanRelativeError(aReferenceVector, aTestVector):
    return np.abs(np.divide(np.subtract(aReferenceVector, aTestVector), aReferenceVector)).mean();

def getMaxRelativeError(aReferenceVector, aTestVector):
    return np.abs(np.divide(np.subtract(aReferenceVector, aTestVector), aReferenceVector)).max();

def getSSIM(aReferenceVector, aTestVector):
    return measure.compare_ssim( aReferenceVector, aTestVector);

def getMSE(aReferenceVector, aTestVector):
    return measure.compare_mse( aReferenceVector, aTestVector);

def getRMSE(aReferenceVector, aTestVector):
    return math.sqrt(getMSE(aReferenceVector, aTestVector));

def getNRMSE_euclidean(aReferenceVector, aTestVector):
    return measure.compare_nrmse(aReferenceVector, aTestVector, 'Euclidean');

def getNRMSE_mean(aReferenceVector, aTestVector):
    return measure.compare_nrmse(aReferenceVector, aTestVector, 'mean');

def getNRMSE_minMax(aReferenceVector, aTestVector):
    return measure.compare_nrmse(aReferenceVector, aTestVector, 'min-max');

def getPSNR(aReferenceVector, aTestVector):
    return measure.compare_psnr(aReferenceVector, aTestVector, aReferenceVector.max() - aReferenceVector.min());

def getNCC(aReferenceVector, aTestVector):
    return productImage(zeroMeanNormalisation(aReferenceVector), zeroMeanNormalisation(aTestVector)).mean();