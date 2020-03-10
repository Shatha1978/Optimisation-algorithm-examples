#!/bin/bash

rm -f *.csv

METRICS=SSE

function extractDataFromLog {

    input_file=$1
    output_file1=$2

    sed 's/, root - INFO - /, root - INFO,/g' $input_file | grep ", root - INFO," - | head -n -1 - > $output_file1
}

function extractLastLine {

    input_file=$1
    output_file1=$2
    output_file2=$3

    # Create the header
    if [ ! -f $output_file2 ]
    then
        STRING=`head -n1 $output_file1`
        echo $STRING > $output_file2
    fi

    # Get the best global fitness
    BEST_FITNESS=`tail -n +2 $output_file1 | cut -d',' -f 8 - | sort - | tail -1`
    LINE_NUMBER=`grep -rne $BEST_FITNESS $output_file1 | cut -f1 -d: - | head -1 -`

    # Extract the line
    sed -n "$LINE_NUMBER,${LINE_NUMBER}p" $output_file1 >> $output_file2
}


function process {

    MODE=$1
    SELECTION=$2
    PREFIX=$3

    # Using mitosis
    input_file=$PREFIX-$MODE-$SELECTION-$run/lamp_problem-$run.log
    output_file1=$PREFIX-$MODE-$SELECTION-$run/lamp_problem-$run.csv
    output_file2=$PREFIX-$MODE-$SELECTION.csv

    rm -f $output_file1

    extractDataFromLog $input_file $output_file1
    extractLastLine $input_file $output_file1 $output_file2
}





for SELECTION in threshold tournament
do
    for MODE in generational steady_state
    do
        rm -f "RUN-FA"-$MODE-$SELECTION.csv

        for run in {1..101}
        do
            process $MODE $SELECTION "RUN-FA"
        done
    done
done
