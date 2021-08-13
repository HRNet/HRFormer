#!/usr/bin/env bash

. config.profile


# PYTHON="python"
PYTHON="/data/anaconda/envs/pytorch1.7.1/bin/python"

# check the enviroment info
export PYTHONPATH="$PWD":$PYTHONPATH
DATA_DIR="${DATA_ROOT}/cityscapes"
BACKBONE="hrt78_win5"
MODEL_NAME="hrt_w78_ocr_v2"
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

# win5:  [1040.3456280679998], time: [0.4938585996627808], params: [56011040]
# win9:  [1064.383171876], time: [0.2658221960067749], params: [56061792]
# win11: [1069.619142052], time: [0.26983258724212644], params: [56098880]
# win15: [1119.8634440679998], time: [0.2960237741470337], params: [56196480]
# msw_hrt78_v1: [1063.5940612839997], time: [0.29960753917694094], params: [56116320]

