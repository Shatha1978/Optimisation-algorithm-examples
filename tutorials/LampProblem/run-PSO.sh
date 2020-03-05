#!/bin/sh
rm -f reconstruction.log

date
for run in {1..15}
do

    OUTPUT_DIR=RUN-PSO-$run
    rm -rf $OUTPUT_DIR
    mkdir $OUTPUT_DIR

    ./LampProblemPSO.py \
        --radius 8 \
        --room_width 100 \
        --room_height 50 \
        --output $OUTPUT_DIR/with_bad_flies-$run \
        --iterations 50 \
        --swarm_size 25 \
        --number_of_lamps 40 \
        --max_stagnation_counter 0 \
        --weight 1 \
        --logging $OUTPUT_DIR/lamp_problem-$run.log

        if [ $? -ne 0 ]; then
            echo There was an error. Stop here.
            exit
        fi
done
date
