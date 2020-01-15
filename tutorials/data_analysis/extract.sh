#!/bin/bash

rm -f *.csv

METRICS=RMSE

for run in {1..15}
do
    for MODE in steady_state generational
    do
        for SELECTION in threshold tournament
        do

            # Using mitosis
            input_file=RUN-$run/reconstruction-$METRICS-$MODE-$SELECTION-mitosis-$run.log
            output_file1=RUN-$run/reconstruction-$METRICS-$MODE-$SELECTION-mitosis-$run.csv
            output_file2=reconstruction-$METRICS-$MODE-$SELECTION-mitosis.csv

            sed 's/, root - INFO - /, root - INFO,/g' $input_file | grep ", root - INFO," - | head -n -1 - > $output_file1

            if [ ! -f $output_file2 ]
            then
                head -n1 $output_file1 > $output_file2
            fi

            tail -1 $output_file1 >> $output_file2


            # Without mitosis
            input_file=RUN-no_mitosis-$run/reconstruction-$METRICS-$MODE-$SELECTION-no_mitosis-$run.log
            output_file1=RUN-no_mitosis-$run/reconstruction-$METRICS-$MODE-$SELECTION-no_mitosis-$run.csv
            output_file2=reconstruction-$METRICS-$MODE-$SELECTION-no_mitosis.csv

            sed 's/, root - INFO - /, root - INFO,/g' $input_file | grep ", root - INFO," - | head -n -1 - > $output_file1

            if [ ! -f $output_file2 ]
            then
                head -n1 $output_file1 > $output_file2
            fi

            tail -1 $output_file1 >> $output_file2

        done
    done
done
