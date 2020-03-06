#!/bin/sh
rm -f reconstruction.log

date
for MODE in steady_state generational
do
    for SELECTION in threshold tournament
    do
        echo "********************************************"
        echo "* $MODE   /   $SELECTION"
        echo "********************************************"
        echo
        for run in {1..15}
        do

            OUTPUT_DIR=RUN-FA-$MODE-$SELECTION-$run
            rm -rf $OUTPUT_DIR
            mkdir $OUTPUT_DIR

            ./LampProblemFA.py \
                --$MODE \
                --radius 8 \
                --room_width 142 \
                --room_height 142 \
                --output $OUTPUT_DIR/with_bad_flies-$run \
                --generations 500 \
                --number_of_lamps 5 \
                --max_stagnation_counter 5 \
                --initial_mutation_variance 16 \
                --final_mutation_variance 8 \
                --weight 1 \
                --selection $SELECTION \
                --tournament_size 2 \
                --visualisation \
                --logging $OUTPUT_DIR/lamp_problem-$run.log
exit
            if [ $? -ne 0 ]; then
                echo There was an error. Stop here.
                exit
            fi

        done

        echo
    done
done
date
