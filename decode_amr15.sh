#!/usr/bin/env bash

# We use the last model to decode (the 84th checkpoint) rather the best model (in terms of PPL)
python3 -m sockeye.translate -m sockeye/amr2015_model \
        --edge-vocab sockeye/data/amr_2015/edge_vocab.json < sockeye/data/amr_2015/test.amrgrh \
        -o sockeye/data/amr_2015/test.snt.out \
        --beam-size 10 \
        --checkpoints 84
