#!/bin/bash

dataset_root_path="/mnt/HDD-1/harsh/datasets/mirflickr/output"

echo "Starting eigen experiments"

make -B
# use taskset to bind to single cpu, eg: taskset -c 7 ./gtnn_stream -n mirflickr -q $dataset_root_path/Q.txt -j streaming_runbook.json -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 999993 -Q 10000 -D 1000
taskset -c 6 ./gtnn_stream -n mirflickr -q $dataset_root_path/Q.txt -j streaming_runbook.json -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 999993 -Q 10000 -D 1000
taskset -c 6 ./gtnn_max -n mirflickr -q $dataset_root_path/Q.txt -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 999993 -Q 10000 -D 1000
taskset -c 6 ./gtnn_sum -n mirflickr -q $dataset_root_path/Q.txt -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 999993 -Q 10000 -D 1000
taskset -c 6 ./dbgtnn_sum -n mirflickr -q $dataset_root_path/Q.txt -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 999993 -Q 10000 -D 1000
