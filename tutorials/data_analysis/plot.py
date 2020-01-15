#!/usr/bin/env python3

import os, fnmatch
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

from ImageMetrics import getNCC, getSSIM

def indexOfMedian(aColumn):
    return df[aColumn][df[aColumn] == df[aColumn].median()].index.tolist();

# Get all the files
#files = fnmatch.filter(os.listdir('.'), '*.csv')

files = [
    "reconstruction-RMSE-generational-tournament-mitosis.csv",
    "reconstruction-RMSE-generational-tournament-no_mitosis.csv",
    "reconstruction-RMSE-generational-threshold-mitosis.csv",
    "reconstruction-RMSE-generational-threshold-no_mitosis.csv",

    "reconstruction-RMSE-steady_state-tournament-mitosis.csv",
    "reconstruction-RMSE-steady_state-tournament-no_mitosis.csv",
    "reconstruction-RMSE-steady_state-threshold-mitosis.csv",
    "reconstruction-RMSE-steady_state-threshold-no_mitosis.csv"
]


data_RMSE_sinogram  = [];
data_ZNCC_sinogram  = [];
data_SSIM_sinogram = [];

data_RMSE_reconstruction  = [];
data_ZNCC_reconstruction  = [];
data_SSIM_reconstruction = [];

data_new_individual_counter = [];

xticks_1 = [1, 2, 3, 4, 5, 6, 7 , 8];
xticks_2 = [
    "generational\ntournament\nmitosis",
    "generational\ntournament\nno mitosis",
    "generational\nthreshold\nmitosis",
    "generational\nthreshold\nno mitosis",

    "steady state\ntournament\nmitosis",
    "steady state\ntournament\nno mitosis",
    "steady state\nthreshold\nmitosis",
    "steady state\nthreshold\nno mitosis"
];


fig = plt.figure(figsize=(12, 5));
plt.axis('off')

reference = np.loadtxt("RUN-1/RMSE-steady_state-threshold-mitosis-1-groundtruth.txt");

i = 0;
for file in files:

    print(file);
    df = pd.read_csv(file);

    data_RMSE_sinogram.append(df['RMSE_sinogram']);
    data_ZNCC_sinogram.append(df['ZNCC_sinogram']);
    data_SSIM_sinogram.append(df['SSIM_sinogram']);

    median_index = indexOfMedian("RMSE_sinogram")[-1];
    print(median_index, len(df['RMSE_sinogram']))

    if i == 0:
        file_name1 = "RUN-" + str(median_index + 1) + "/with_bad_flies-RMSE-generational-tournament-mitosis-" + str(median_index + 1) + "-reconstruction.";
        file_name2 = "RUN-" + str(median_index + 1) + "/without_bad_flies-RMSE-generational-tournament-mitosis-" + str(median_index + 1) + "-reconstruction.";
    elif i == 1:
        file_name1 = "RUN-no_mitosis-" + str(median_index + 1) + "/with_bad_flies-RMSE-generational-tournament-no_mitosis-" + str(median_index + 1) + "-reconstruction.";
        file_name2 = "RUN-no_mitosis-" + str(median_index + 1) + "/without_bad_flies-RMSE-generational-tournament-no_mitosis-" + str(median_index + 1) + "-reconstruction.";
    elif i == 2:
        file_name1 = "RUN-" + str(median_index + 1) + "/with_bad_flies-RMSE-generational-threshold-mitosis-" + str(median_index + 1) + "-reconstruction.";
        file_name2 = "RUN-" + str(median_index + 1) + "/without_bad_flies-RMSE-generational-threshold-mitosis-" + str(median_index + 1) + "-reconstruction.";
    elif i == 3:
        file_name1 = "RUN-no_mitosis-" + str(median_index + 1) + "/with_bad_flies-RMSE-generational-threshold-no_mitosis-" + str(median_index + 1) + "-reconstruction.";
        file_name2 = "RUN-no_mitosis-" + str(median_index + 1) + "/without_bad_flies-RMSE-generational-threshold-no_mitosis-" + str(median_index + 1) + "-reconstruction.";
    elif i == 4:
        file_name1 = "RUN-" + str(median_index + 1) + "/with_bad_flies-RMSE-steady_state-tournament-mitosis-" + str(median_index + 1) + "-reconstruction.";
        file_name2 = "RUN-" + str(median_index + 1) + "/without_bad_flies-RMSE-steady_state-tournament-mitosis-" + str(median_index + 1) + "-reconstruction.";
    elif i == 5:
        file_name1 = "RUN-no_mitosis-" + str(median_index + 1) + "/with_bad_flies-RMSE-steady_state-tournament-no_mitosis-" + str(median_index + 1) + "-reconstruction.";
        file_name2 = "RUN-no_mitosis-" + str(median_index + 1) + "/without_bad_flies-RMSE-steady_state-tournament-no_mitosis-" + str(median_index + 1) + "-reconstruction.";
    elif i == 6:
        file_name1 = "RUN-" + str(median_index + 1) + "/with_bad_flies-RMSE-steady_state-threshold-mitosis-" + str(median_index + 1) + "-reconstruction.";
        file_name2 = "RUN-" + str(median_index + 1) + "/without_bad_flies-RMSE-steady_state-threshold-mitosis-" + str(median_index + 1) + "-reconstruction.";
    elif i == 7:
        file_name1 = "RUN-no_mitosis-" + str(median_index + 1) + "/with_bad_flies-RMSE-steady_state-threshold-no_mitosis-" + str(median_index + 1) + "-reconstruction.";
        file_name2 = "RUN-no_mitosis-" + str(median_index + 1) + "/without_bad_flies-RMSE-steady_state-threshold-no_mitosis-" + str(median_index + 1) + "-reconstruction.";

    ax = plt.subplot(2, 8, i + 1)
    img = np.loadtxt(file_name1 + "txt")
    ax.imshow(img, cmap='gray')
    ax.set_title(xticks_2[i] + "\nSSIM: " + str(round(100 * getSSIM(reference, img))) + "%");


    ax = plt.subplot(2, 8, i + 1 + 8)
    img = np.loadtxt(file_name2 + "txt")
    ax.imshow(img, cmap='gray')
    ax.set_title("\nSSIM: " + str(round(100 * getSSIM(reference, img))) + "%");


    #print(indexOfMedian("ZNCC_sinogram"))
    #print(indexOfMedian("SSIM_sinogram"))

    data_RMSE_reconstruction.append(df['RMSE_reconstruction']);
    data_ZNCC_reconstruction.append(df['ZNCC_reconstruction']);
    data_SSIM_reconstruction.append(df['SSIM_reconstruction']);

    #print(indexOfMedian("RMSE_reconstruction"))
    #print(indexOfMedian("ZNCC_reconstruction"))
    #print(indexOfMedian("SSIM_reconstruction"))


    data_new_individual_counter.append(df['new_individual_counter']);
    i += 1;
fig.savefig("reconstructions.pdf", bbox_inches='tight')



fig = plt.figure(figsize=(10, 5));
plt.title('RMSE sinogram (global fitness)')
plt.boxplot(data_RMSE_sinogram)
plt.xticks(xticks_1, xticks_2);
fig.savefig("RMSE_sinogram.pdf", bbox_inches='tight')

fig = plt.figure(figsize=(10, 5));
plt.title('ZNCC sinogram')
plt.boxplot(data_ZNCC_sinogram)
plt.xticks(xticks_1, xticks_2);
fig.savefig("ZNCC_sinogram.pdf", bbox_inches='tight')

fig = plt.figure(figsize=(10, 5));
plt.title('SSIM sinogram')
plt.boxplot(data_SSIM_sinogram)
plt.xticks(xticks_1, xticks_2);
fig.savefig("SSIM_sinogram.pdf", bbox_inches='tight')

fig = plt.figure(figsize=(10, 5));
plt.title('RMSE reconstruction')
plt.boxplot(data_RMSE_reconstruction)
plt.xticks(xticks_1, xticks_2);
fig.savefig("RMSE_reconstruction.pdf", bbox_inches='tight')

fig = plt.figure(figsize=(10, 5));
plt.title('ZNCC reconstruction')
plt.boxplot(data_ZNCC_reconstruction)
plt.xticks(xticks_1, xticks_2);
fig.savefig("ZNCC_reconstruction.pdf", bbox_inches='tight')

fig = plt.figure(figsize=(10, 5));
plt.title('SSIM reconstruction')
plt.boxplot(data_SSIM_reconstruction)
plt.xticks(xticks_1, xticks_2);
fig.savefig("SSIM_reconstruction.pdf", bbox_inches='tight')

fig = plt.figure(figsize=(10, 5));
plt.title('Number of individuals created')
plt.boxplot(data_new_individual_counter)
plt.xticks(xticks_1, xticks_2);
fig.savefig("new_individual_counter.pdf", bbox_inches='tight')

plt.show()
