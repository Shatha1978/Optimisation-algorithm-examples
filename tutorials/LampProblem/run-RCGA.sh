#!/bin/sh
rm -f reconstruction.log

date
for run in {1..15}
do

    for SELECTION in roulette tournament  ranking
    do

        OUTPUT_DIR=RUN-RCGA-$SELECTION-$run
        rm -rf $OUTPUT_DIR
        mkdir $OUTPUT_DIR

        ./LampProblemRCGA.py \
            --radius 8 \
            --room_width 150 \
            --room_height 150 \
            --output $OUTPUT_DIR/with_bad_flies-$run \
            --generations 50 \
            --pop_size 50 \
            --number_of_lamps 50 \
            --max_stagnation_counter 0 \
            --initial_mutation_variance 5 \
            --final_mutation_variance 2 \
            --weight 0.25 \
            --selection $SELECTION \
            --tournament_size 2 \
            --logging $OUTPUT_DIR/lamp_problem-$run.log \
            --visualisation

        if [ $? -ne 0 ]; then
            echo There was an error. Stop here.
            exit
        fi

    done
done
date
