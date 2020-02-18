#!/bin/sh

METRICS=SSE

date
for run in {1..15}
do

    OUTPUT_DIR=RUN-no_mitosis-$run
    rm -rf $OUTPUT_DIR
    mkdir $OUTPUT_DIR

    for MODE in steady_state generational
    do
        for SELECTION in threshold tournament
        do
            PREFIX=$METRICS-$MODE-$SELECTION-no_mitosis

            ./TomographyReconstruction.py \
                --angles 25 \
                --save_input_images $OUTPUT_DIR/$PREFIX-$run \
                --output_without_bad_flies $OUTPUT_DIR/without_bad_flies-$PREFIX-$run \
                --output_with_bad_flies $OUTPUT_DIR/with_bad_flies-$PREFIX-$run \
                --peak 50 \
                --generations 250 \
                --initial_pop_size 1840 \
                --max_pop_size 1840 \
                --input derenzo-hot.png \
                --max_stagnation_counter 5 \
                --initial_lambda -0.01 \
                --final_lambda -0.25 \
                --initial_mutation_variance 5 \
                --final_mutation_variance 2 \
                --initial_new_blood_probability 0.25 \
                --final_new_blood_probability 0.05 \
                --selection $SELECTION \
                --tournament_size 2 \
                --$MODE \
                --logging $OUTPUT_DIR/reconstruction-$PREFIX-$run.log \
                --objective $METRICS

                if [ $? -ne 0 ]; then
                    echo There was an error. Stop here.
                    exit
                fi
        done
    done
done
date
