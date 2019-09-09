#!/usr/bin/env bash
python3 -m sockeye.translate -m sockeye/en2de_model \
        --edge-vocab sockeye/data/en2de/edge_vocab.json < sockeye/data/en2de/test.en.tokdeps \
        -o sockeye/data/en2de/test.mt.out \
        --beam-size 10