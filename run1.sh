#!/bin/bash

results_dir="results/20250424"
log_dir="logs/20250424"

# Create the results directory if it doesn't exist
if [ ! -d "$results_dir" ]; then
    mkdir -p $results_dir
fi

if [ ! -d "$log_dir" ]; then
    mkdir -p $log_dir
fi

# Function to run the Python commands and log output
run_and_log() {
    dataset=$1
    log_file="$log_dir/$dataset.log"
    # create the log file
    touch $log_file

    echo "Running for dataset: $dataset" # >> $log_file
    echo "===================="          # >> $log_file
    python main.py --dataset "$dataset" --results_dir "$results_dir" >>$log_file 2>&1
    python evaluation_test.py --dataset "$dataset" --results_dir "$results_dir" >>$log_file 2>&1
    echo "Finished for dataset: $dataset" # >> $log_file
    echo "===================="           # >> $log_file

    # # Split the log file into smaller chunks (e.g., 10MB each)
    # split -b 10M $log_file ${log_file}_part_

    # # Optional: Remove the original log file after splitting
    # rm $log_file
}

# Run the commands for each dataset
# run_and_log "chd_49"
# run_and_log "emotions"
# run_and_log "VirusPseAAC"
# run_and_log "GpositivePseAAC"
run_and_log "PlantPseAAC"
run_and_log "water-quality"
run_and_log "scene"
# run_and_log "yeast"
run_and_log "HumanPseAAC"
