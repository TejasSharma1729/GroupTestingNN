#!/bin/bash

dataset_root_path="/mnt/HDD-1/harsh/datasets/insta_1m/output"

echo "Starting eigen experiments"

make -B
# use taskset to bind to single cpu, eg: taskset -c 7 ./gtnn_stream -n insta_1m -q $dataset_root_path/Q.txt -j streaming_runbook.json -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 999991 -Q 10000 -D 1000
taskset -c 4 ./gtnn_stream -n insta_1m -q $dataset_root_path/Q.txt -j streaming_runbook.json -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 999991 -Q 10000 -D 1000
taskset -c 4 ./gtnn_max -n insta_1m -q $dataset_root_path/Q.txt -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 999991 -Q 10000 -D 1000
taskset -c 4 ./gtnn_sum -n insta_1m -q $dataset_root_path/Q.txt -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 999991 -Q 10000 -D 1000
taskset -c 4 ./dbgtnn_sum -n insta_1m -q $dataset_root_path/Q.txt -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 999991 -Q 10000 -D 1000
