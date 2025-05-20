#!/bin/bash

if [ "$#" -ge 1 ] && [ "$1" == "--no_adarft" ]; then
    task_file=~/verl/examples/adarft/run_qwen2.5-1.5b_seq_balance_no_adarft.sh
else
    task_file=~/verl/examples/adarft/run_qwen2.5-1.5b_seq_balance.sh
fi

bash ${task_file} easy_extreme
bash ${task_file} hard_extreme
bash ${task_file} skew_difficult
bash ${task_file} skew_easy
bash ${task_file} uniform
