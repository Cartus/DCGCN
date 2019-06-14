#!/usr/bin/env bash
python3 -m sockeye.train --source sockeye/data/silver/mix.amr \
        --target sockeye/data/silver/mix.snt \
        --source-graphs sockeye/data/silver/mix.grh \
        --validation-source sockeye/data/silver/test.amr \
        --validation-target sockeye/data/silver/test.snt \
        --val-source-graphs sockeye/data/silver/test.grh \
        --edge-vocab sockeye/data/silver/edge_vocab.json \
        --batch-size 30 \
        --batch-type sentence \
        --word-min-count 3:3 \
        --num-embed 360:360 \
        --embed-dropout .5:.5 \
        --max-seq-len 169:169 \
        --encoder gcn \
        --gcn-activation relu \
        --gcn-num-hidden 360 \
        --gcn-pos-embed 300 \
        --decoder rnn \
        --num-layers 1:2 \
        --rnn-num-hidden 360 \
        --rnn-decoder-hidden-dropout 0.2 \
        --checkpoint-frequency 1500 \
        --max-num-checkpoint-not-improved 28 \
        --initial-learning-rate 0.0003 \
        --learning-rate-reduce-factor 0.7 \
        --learning-rate-reduce-num-not-improved 5 \
        --gcn-num-layers 4 \
        --weight-init-xavier-factor-type in \
        --weight-init-scale 2.34 \
        --decode-and-evaluate -1 \
        --output sockeye/silver_model \
        --overwrite-output \
        --device-ids 0 \
        --gcn-dropout 0.1 \
        --gcn-adj-norm \
        --rnn-attention-type coverage \
        --shared-vocab \
        --weight-tying \
        --weight-tying-type src_trg
