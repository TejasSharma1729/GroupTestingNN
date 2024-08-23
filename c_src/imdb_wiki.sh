#!/bin/bash

dataset_root_path="/mnt/HDD-1/harsh/datasets/imdb_wiki/output"

echo "Starting eigen experiments"

make -B
# use taskset to bind to single cpu, eg: taskset -c 7 ./gtnn_stream -n imdb_wiki -q $dataset_root_path/Q.txt -j streaming_runbook.json -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 520458 -Q 10000 -D 1000
# taskset -c 5 ./gtnn_stream -n imdb_wiki -q $dataset_root_path/Q.txt -j streaming_runbook.json -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 520458 -Q 10000 -D 1000
# taskset -c 5 ./gtnn_max -n imdb_wiki -q $dataset_root_path/Q.txt -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 520458 -Q 10000 -D 1000
# taskset -c 5 ./gtnn_sum -n imdb_wiki -q $dataset_root_path/Q.txt -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 520458 -Q 10000 -D 1000
# taskset -c 5 ./dbgtnn_sum -n imdb_wiki -q $dataset_root_path/Q.txt -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 520458 -Q 10000 -D 1000
taskset -c 5 ./classwise_dbgtnn_sum -n imdb_wiki -q $dataset_root_path/Q.txt -d $dataset_root_path/X.txt -r ./results -R 0.8 -N 520458 -Q 10000 -D 1000
