#!/usr/bin/env bash

MODEL=$1
DATAPATH=$2
DATASET=$3

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port 12348 \
    test_flops.py \
    --mode flops    \
    --data-path $DATAPATH \
    --fig_num 100 \
    --data-set $DATASET  \
    --cfg configs/$MODEL.yaml \
    --amp-opt-level O1
