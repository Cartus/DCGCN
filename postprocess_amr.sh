#!/usr/bin/env bash

# Change the directory here
DATA_DIR=./sockeye/data/amr_2015

python3 ./sockeye/postprocess.py ${DATA_DIR}/map.pp.txt ${DATA_DIR}/test.snt.out ${DATA_DIR}/final.txt
