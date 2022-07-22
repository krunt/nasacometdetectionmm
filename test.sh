#!/bin/bash

testdatapath=$1
outputcsvpath=$2

cachepath=/wdata/test
modeldir=/root/modeldir
precsvpath=/wdata/solution.pre.csv

echo "Test" $testdatapath

rm -rf $cachepath
mkdir $cachepath


python3 preprocess.py $testdatapath $cachepath

python3 test.py $testdatapath $cachepath $modeldir $precsvpath

python3 postprocess.py $testdatapath $cachepath $precsvpath $outputcsvpath

