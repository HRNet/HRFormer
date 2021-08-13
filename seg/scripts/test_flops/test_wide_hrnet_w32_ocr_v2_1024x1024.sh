#!/usr/bin/env bash

. config.profile


PYTHON="python"

# check the enviroment info
export PYTHONPATH="$PWD":$PYTHONPATH
DATA_DIR="${DATA_ROOT}/cityscapes"
BACKBONE="wide_hrnet32"
MODEL_NAME="wide_hrnet_w32_ocr_v2"
CONFIGS="configs/cityscapes/H_48_D_4.json"


CUDA_VISIBLE_DEVICES=7 $PYTHON -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port 12348 \
    main.py \
    --phase test_flops \
    --configs ${CONFIGS} --drop_last y \
    --backbone ${BACKBONE} --model_name ${MODEL_NAME} \
    --val_batch_size 1 \
    --shape 1024 1024 \
    --test_dir ${DATA_DIR}/val/image \
    --data_dir ${DATA_DIR} 
