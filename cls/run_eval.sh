#!/usr/bin/env bash
MODEL=$1
RESUME=$2
DATAPATH=$3

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                    --nproc_per_node=4 \
                                    --master_port 12345 \
                                    main.py \
                                    --eval \
                                    --zip \
                                    --cfg configs/$MODEL.yaml \
                                    --resume $RESUME \
                                    --data-path $DATAPATH
