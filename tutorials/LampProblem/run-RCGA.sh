#!/bin/sh
rm -f reconstruction.log

date
for run in {1..15}
do

    for SELECTION in roulette tournament  ranking
    do

        echo "********************************************"
        echo "* $SELECTION"
        echo "********************************************"
        echo

        OUTPUT_DIR=RUN-RCGA-$SELECTION-$run
        rm -rf $OUTPUT_DIR
        mkdir $OUTPUT_DIR

        ./LampProblemRCGA.py \
            --radius 8 \
            --room_width 100 \
            --room_height 50 \
            --output $OUTPUT_DIR/with_bad_flies-$run \
            --generations 50 \
            --pop_size 25 \
            --number_of_lamps 40 \
            --max_stagnation_counter 0 \
            --initial_mutation_variance 16 \
            --final_mutation_variance 8 \
            --weight 1 \
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
