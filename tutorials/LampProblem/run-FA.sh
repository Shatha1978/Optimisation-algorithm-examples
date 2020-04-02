#!/bin/sh

date
for run in {1..101}
do
    for pb_size in 3 5 10 20 100 500
    do

        echo "********************************************"
        echo "* PROBLEM SIZE: $pb_size"
        echo "********************************************"
        echo

        pd_dir="pb_size_$pb_size"
        if [ ! -d $pd_dir ]
        then
            echo Create $pd_dir
            mkdir $pd_dir
        fi

        for MODE in generational steady_state
        do
            for SELECTION in tournament threshold
            do

                echo "********************************************"
                echo "* $MODE/$SELECTION/$run"
                echo "********************************************"
                echo

                nb_lamps=`echo "3*$pb_size"|bc`

                case $pb_size in

                    3)
                        room_size=25
                        pop_size=`echo "$pb_size"`
                        ;;

                    5)
                        room_size=32
                        pop_size=`echo "$pb_size"`
                        ;;

                    10)
                        room_size=45
                        pop_size=`echo "$pb_size"`
                        ;;

                    20)
                        room_size=63
                        pop_size=`echo "$pb_size"`
                        ;;

                    100)
                        room_size=142
                        pop_size=`echo "$pb_size"`
                        ;;

                    500)
                        room_size=317
                        pop_size=`echo "$pb_size"`
                        ;;

                    1000)
                        room_size=448
                        pop_size=`echo "$pb_size"`
                        ;;

                    *)
                        echo "$pb_size is an invalid problem size"
                        exit
                        ;;
                esac

                OUTPUT_DIR=$pd_dir/RUN-FA-$MODE-$SELECTION-$run
                rm -rf $OUTPUT_DIR

                echo "mkdir $OUTPUT_DIR"
                mkdir $OUTPUT_DIR

                ./LampProblemFA.py \
                    --$MODE \
                    --radius 8 \
                    --room_width $room_size \
                    --room_height $room_size \
                    --output $OUTPUT_DIR/enlightment-$run \
                    --generations 500 \
                    --number_of_lamps $pop_size \
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
        done
    done
done
date
