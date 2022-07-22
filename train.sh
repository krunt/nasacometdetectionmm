#!/bin/bash


traindatapath=$1
cachepath=/wdata/train
modeldir=/root/modeldir

echo "Train" $traindatapath

rm -rf $cachepath
mkdir $cachepath

# removing models
rm -f $modeldir/*


python3 preprocess.py $traindatapath $cachepath


python3 train.py $traindatapath $cachepath $modeldir

