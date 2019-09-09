#!/usr/bin/env bash

# For AMR-to-Text
python3 -m sockeye.evaluate -r sockeye/data/amr_2015/surface.pp.txt  -i sockeye/data/amr_2015/final.txt

# For NMT
#python3 -m sockeye.evaluate -r sockeye/data/en2de/test.de.tok  -i sockeye/data/en2de/test.mt.out