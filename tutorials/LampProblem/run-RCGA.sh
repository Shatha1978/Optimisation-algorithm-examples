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

        for SELECTION in roulette tournament  ranking
        do

            echo "********************************************"
            echo "* $SELECTION/$run"
            echo "********************************************"
            echo

            nb_lamps=`echo "3*$pb_size"|bc`

            case $pb_size in

                3)
                    room_size=25
                    pop_size=`echo "sqrt(128 / $nb_lamps)" | bc`
                    ;;

                5)
                    room_size=32
                    pop_size=`echo "sqrt(254 / $nb_lamps)" | bc`
                    ;;

                10)
                    room_size=45
                    pop_size=`echo "sqrt(504 / $nb_lamps)" | bc`
                    ;;

                20)
                    room_size=63
                    pop_size=`echo "sqrt(950 / $nb_lamps)" | bc`
                    ;;

                100)
                    room_size=142
                    pop_size=`echo "sqrt(4105 / $nb_lamps)" | bc`
                    ;;

                500)
                    room_size=317
                    pop_size=`echo "sqrt(22186 / $nb_lamps)" | bc`
                    ;;

                1000)
                    exit
                    room_size=448
                    pop_size=`echo "sqrt(??? / $nb_lamps)" | bc`
                    ;;

                *)
                    echo "$pb_size is an invalid problem size"
                    exit
                    ;;
            esac

            OUTPUT_DIR=$pd_dir/RUN-RCGA-$SELECTION-$run
            rm -rf $OUTPUT_DIR
            mkdir $OUTPUT_DIR

            ./LampProblemRCGA.py \
                --radius 8 \
                --room_width $room_size \
                --room_height $room_size \
                --output $OUTPUT_DIR/enlightment-$run \
                --generations 500 \
                --pop_size $pop_size \
                --number_of_lamps $nb_lamps \
                --max_stagnation_counter 50 \
                --initial_mutation_variance 16 \
                --final_mutation_variance 8 \
                --weight 1 \
                --selection $SELECTION \
                --tournament_size 2 \
                --logging $OUTPUT_DIR/lamp_problem-$run.log > /dev/null 2> /dev/null

            if [ $? -ne 0 ]; then
                echo There was an error. Stop here.
                exit
            fi
        done
    done
done
date
