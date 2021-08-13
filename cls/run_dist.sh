#!/usr/bin/env bash
MODEL=$1
DATAPATH=$2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
                                    --nproc_per_node=8 \
                                    --master_port 12345 \
                                    main.py \
                                    --cfg configs/$MODEL.yaml \
                                    --batch-size 128 \
                                    --zip \
                                    --data-path $DATAPATH \
                                    --output_dir "output/$MODEL" \
                                    --tag $(date +%F_%H-%M-%S)