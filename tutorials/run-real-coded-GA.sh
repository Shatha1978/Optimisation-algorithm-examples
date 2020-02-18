#!/bin/sh
rm -f reconstruction.log

METRICS=SSE

date
for run in {1..15}
do

    for SELECTION in tournament roulette ranking
    do

        OUTPUT_DIR=RUN-GA-$SELECTION-$run
        rm -rf $OUTPUT_DIR
        mkdir $OUTPUT_DIR

        ./TomographyReconstructionRCGA.py \
            --angles 25 \
            --save_input_images $OUTPUT_DIR/$METRICS-$SELECTION-$run \
            --output $OUTPUT_DIR/without_bad_flies-$METRICS-$SELECTION-$run \
            --peak 50 \
            --generations 142 \
            --pop_size 142 \
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
    done
done
date
