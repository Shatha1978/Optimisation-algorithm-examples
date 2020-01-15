#!/bin/sh
rm -f reconstruction.log

SELECTION=tournament
SELECTION=threshold
MODE=steady_state
MODE=generational
METRICS=SSE

date
for run in {1..15}
do

    OUTPUT_DIR=RUN-GA-$run
    rm -rf $OUTPUT_DIR
    mkdir $OUTPUT_DIR

    for SELECTION in tournament roulette ranking
    do

        ./TomographyReconstruction1.py \
            --angles 25 \
            --save_input_images $OUTPUT_DIR/$METRICS-$SELECTION-$run \
            --output $OUTPUT_DIR/without_bad_flies-$METRICS-$SELECTION-$run \
            --peak 50 \
            --generations 166 \
            --pop_size 166 \
            --number_of_emission_points 1840 \
            --input derenzo-hot.png \
            --max_stagnation_counter 5 \
            --initial_lambda -0.01 \
            --final_lambda -0.25 \
            --initial_mutation_variance 5 \
            --final_mutation_variance 2 \
            --selection $SELECTION \
            --tournament_size 2 \
            --logging $OUTPUT_DIR/reconstruction-$METRICS-$SELECTION-$run.log \
            --objective $METRICS


            if [ $? -ne 0 ]; then
                echo There was an error. Stop here.
                exit
            fi

            #grep "root - INFO, " $OUTPUT_DIR/reconstruction-$METRICS-$SELECTION-$run.log > $OUTPUT_DIR/reconstruction-$METRICS-$SELECTION-$run.csv

            #clear;./parCoord.py reconstruction--$METRICS-$SELECTION.csv
    done
done
date
