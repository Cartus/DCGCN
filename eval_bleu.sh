#!/usr/bin/env bash

#Change the data directory here (amr_2015) for other datasets (amr_2017)
python3 -m sockeye.evaluate -r sockeye/data/amr_2015/surface.pp.txt  -i sockeye/data/amr_2015/final.txt