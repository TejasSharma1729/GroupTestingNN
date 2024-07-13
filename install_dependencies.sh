#!/bin/bash

set -e

cd c_src

## setting up eigen 
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip

## setting up json reader for c++
wget https://github.com/nlohmann/json.git 


## python libs
conda create --name ann --file environment.yml
conda activate ann