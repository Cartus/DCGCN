#!/usr/bin/env bash
python3 -m sockeye.translate -m sockeye/amr_model \
        --edge-vocab sockeye/data/gold/edge_vocab.json < sockeye/data/gold/test.amrgrh \
        -o sockeye/data/gold/test.snt.out \
        --beam-size 10
        #--checkpoints 136
