#!/bin/sh
rm -f reconstruction.log

date
for run in {1..15}
do

    for SELECTION in tournament roulette ranking
    do

        OUTPUT_DIR=RUN-RCGA-$SELECTION-$run
        rm -rf $OUTPUT_DIR
        mkdir $OUTPUT_DIR

        ./LampProblemRCGA.py \
            --radius 5 \
            --room_width 50 \
            --room_height 50 \
            --output $OUTPUT_DIR/with_bad_flies-$run \
            --generations 50 \
            --pop_size 50 \
            --number_of_lamps 25 \
            --max_stagnation_counter 0 \
            --weight 0.5 \
            --selection $SELECTION \
            --tournament_size 2 \
            --logging $OUTPUT_DIR/lamp_problem-$run.log

        if [ $? -ne 0 ]; then
            echo There was an error. Stop here.
            exit
        fi

    done
done
date
