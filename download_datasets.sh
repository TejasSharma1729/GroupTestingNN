#!/bin/bash

set -e

root=./datasets
repopath=./

mkdir -p $root 
cd $root 
mkdir datasets
cd datasets

# MIRFLICKR
mkdir mirflickr
cd mirflickr

mkdir images
mkdir output

for i in {0..9};
do
    wget https://press.liacs.nl/mirflickr/mirflickr1m.v3b/images$i.zip && unzip images$i.zip -d images && rm images$i.zip
done
python $repopath/py_src/utils/vgg16_features.py --root images --save-path output --quite
python $repopath/py_src/utils/create_queries.py --root images --save-path output
rm -rf output/vgg_feature_mat_*

cd ..

# IMDB-WIKI
mkdir imdb_wiki
cd imdb_wiki

mkdir images
mkdir output

for i in {0..9};
do
    wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_$i.tar && tar -C images -xvf imdb_$i.tar && rm imdb_$i.tar
done
wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki.tar.gz && tar -C images -xvzf wiki.tar.gz && rm imdb_$i.tar
python $repopath/py_src/utils/vgg16_features.py --root images --save-path output --quite
python $repopath/py_src/utils/create_queries.py --root images --save-path output
rm -rf output/vgg_feature_mat_*
cd ..

# InstaCities
mkdir insta_1m
cd insta_1m

mkdir images
mkdir output

pip install gdown && gdown 1SCh8gSoyvrJ7N9OwlRcWc65zo1qf6SAy && unzip InstaCities1M.zip -d images && rm InstaCities1M.zip
python $repopath/py_src/utils/vgg16_features.py --root images --save-path output --quite
python $repopath/py_src/utils/create_queries.py --root images --save-path output
rm -rf output/vgg_feature_mat_*
cd ..

echo "Successfully downloaded and setup the datasets"

# Script by Harsh-Sensei