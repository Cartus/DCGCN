#!/usr/bin/env bash
python3 -m sockeye.translate -m sockeye/amr2015_model \
        --edge-vocab sockeye/data/amr_2015/edge_vocab.json < sockeye/data/amr_2015/test.amrgrh \
        -o sockeye/data/amr_2015/test.snt.out \
        --beam-size 10 \
        --checkpoints 103
