#!/usr/bin/env bash

. config.profile

PYTHON="python"
PYTHON="/data/anaconda/envs/pytorch1.7.1/bin/python"

# check the enviroment info
export PYTHONPATH="$PWD":$PYTHONPATH
DATA_DIR="${DATA_ROOT}/cityscapes"
BACKBONE="hrt32_win11"
MODEL_NAME="hrt_w32_ocr_v2"
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

# win9: [422.8918078339999], time: [0.18235397338867188], params: [13538496]
# win11: [426.503305604], time: [0.1842280626296997], params: [13557040]

