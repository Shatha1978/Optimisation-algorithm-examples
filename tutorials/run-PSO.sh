#!/bin/sh
rm -f reconstruction.log

METRICS=SSE

date
for run in {1..15}
do

    OUTPUT_DIR=RUN-PSO-$run
    rm -rf $OUTPUT_DIR
    mkdir $OUTPUT_DIR

    ./TomographyReconstructionPSO.py \
        --angles 25 \
        --save_input_images $OUTPUT_DIR/$METRICS-$run \
        --output $OUTPUT_DIR/without_bad_flies-$METRICS-$run \
        --peak 50 \
        --iterations 142 \
        --swarm_size 142 \
        --number_of_emission_points 1840 \
        --input derenzo-hot.png \
        --max_stagnation_counter 0 \
        --initial_lambda -0.01 \
        --final_lambda -0.25 \
        --logging $OUTPUT_DIR/reconstruction-$METRICS-$run.log \
        --objective $METRICS


        if [ $? -ne 0 ]; then
            echo There was an error. Stop here.
            exit
        fi
done
date

