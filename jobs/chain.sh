#!/bin/bash

# Define the number of tasks in the chain and the prefix for job naming
ARRAY_SIZE=5  # Number of tasks
JOB_SCRIPT="/lustre/orion/bif151/scratch/istabrak/new/continual_neox/gpt-neox/job.sh"

# Define the prefixes for different configurations, if any (you can modify this to suit your needs)
prefixes=("config_1" "config_2" "config_2" "config_2" "config_2")

# Loop through each prefix (config) and submit jobs in a chain
for prefix in "${prefixes[@]}"; do
    echo "Submitting jobs for prefix: $prefix"

    # Submit the first job in the chain
    JOB_ID=$(sbatch --job-name=${prefix}_task1 ${JOB_SCRIPT} ${prefix} | awk '{print $4}')
    echo "Submitted job ${prefix}_task1 with JOB_ID: $JOB_ID"

    # Submit the remaining jobs with a dependency on the previous job
    for TASK_ID in $(seq 2 $ARRAY_SIZE); do
        JOB_ID=$(sbatch --dependency=afterany:${JOB_ID} --job-name=${prefix}_task${TASK_ID} ${JOB_SCRIPT} ${prefix} | awk '{print $4}')
        echo "Submitted job ${prefix}_task${TASK_ID} with JOB_ID: $JOB_ID"
    done
done
