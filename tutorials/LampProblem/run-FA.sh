#!/bin/sh

current_dir=$PWD;

date
for pb_size in 1000 3 5 10 20 100
do
    # Restore the working directory
    cd $current_dir

    pd_dir="pb_size_$pb_size"
    if [ ! -d $pd_dir ]
    then
        echo Create $pd_dir
        mkdir $pd_dir
    fi

    for MODE in generational steady_state
    do
        for SELECTION in threshold tournament
        do
            echo "********************************************"
            echo "* $MODE   /   $SELECTION"
            echo "********************************************"
            echo
            for run in {1..101}
            do

                case $pb_size in

                    3)
                        room_size=25
                        ;;

                    5)
                        room_size=32
                        ;;

                    10)
                        room_size=45
                        ;;

                    20)
                        room_size=63
                        ;;

                    100)
                        room_size=142
                        ;;

                    1000)
                        room_size=317
                        ;;

                    *)
                        echo "$pb_size is an invalid problem size"
                        exit
                        ;;
                esac

                OUTPUT_DIR=$pd_dir/RUN-FA-$MODE-$SELECTION-$run
                rm -rf $OUTPUT_DIR
                mkdir $OUTPUT_DIR

                ./LampProblemFA.py \
                    --$MODE \
                    --radius 8 \
                    --room_width $room_size \
                    --room_height $room_size \
                    --output $OUTPUT_DIR/enlightment-$run \
                    --generations 500 \
                    --number_of_lamps 5 \
                    --max_stagnation_counter 5 \
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

            echo
        done
    done

    # Change the working directory
    cd $current_dir/$pb_size
    ../extact.sh
done

# Restore the working directory
cd $current_dir
./boxplot.py

date
