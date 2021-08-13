#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
cd ../../../
. config.profile

# check the enviroment info
nvidia-smi
${PYTHON} -m pip install yacs
${PYTHON} -m pip install torchcontrib
${PYTHON} -m pip install git+https://github.com/lucasb-eyer/pydensecrf.git

${PYTHON} -m pip install timm
${PYTHON} -m pip install einops

export PYTHONPATH="$PWD":$PYTHONPATH

DATA_DIR="${DATA_ROOT}/ade20k"
SAVE_DIR="${DATA_ROOT}/seg_result/ade20k/"
BACKBONE="hrt_base"
CONFIGS="configs/ade20k/H_48_D_4.json"
CONFIGS_TEST="configs/ade20k/H_48_D_4_TEST_MS3x.json"

MODEL_NAME="hrt_base_ocr_v2"
LOSS_TYPE="fs_auxohemce_loss"
CHECKPOINTS_NAME="${MODEL_NAME}_${BACKBONE}_ohem_$(date +%F_%H-%M-%S)"
PRETRAINED_MODEL="./hrt_pretrained_models/hrt_base.pth" # Replace with the path to pre-trained backbone
MAX_ITERS=150000

LOG_FILE="./log/ade20k/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`

OPTIM_METHOD='adamw'
BASE_LR=0.0001
LR_POLICY='warm_lambda_poly'    # linear warmup
GROUP_METHOD='decay'    # no weight decay on norm and pos embed


if [ "$1"x == "train"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} \
                       --drop_last y \
                       --phase train \
                       --gathered n \
                       --loss_balance y \
                       --log_to_file n \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --gpu 0 1 2 3 4 5 6 7 \
                       --data_dir ${DATA_DIR} \
                       --loss_type ${LOSS_TYPE} \
                       --max_iters ${MAX_ITERS} \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --pretrained ${PRETRAINED_MODEL} \
                       --optim_method ${OPTIM_METHOD} \
                       --base_lr ${BASE_LR} \
                       --lr_policy ${LR_POLICY} \
                       --group_method ${GROUP_METHOD} \
                       --distributed \
                       --test_interval 10000 \
                       2>&1 | tee ${LOG_FILE}


elif [ "$1"x == "local"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} \
                       --drop_last y \
                       --phase train \
                       --gathered n \
                       --loss_balance y \
                       --log_to_file n \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --gpu 0 1 2 3 \
                       --data_dir ${DATA_DIR} \
                       --loss_type ${LOSS_TYPE} \
                       --max_iters ${MAX_ITERS} \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --pretrained ${PRETRAINED_MODEL} \
                       --optim_method ${OPTIM_METHOD} \
                       --base_lr ${BASE_LR} \
                       --lr_policy ${LR_POLICY} \
                       --group_method ${GROUP_METHOD} \
                       --distributed \
                       --train_batch_size 8 \
                       --test_interval 10000 \
                       2>&1 | tee ${LOG_FILE}


elif [ "$1"x == "resume"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} \
                       --drop_last y \
                       --phase train \
                       --gathered n \
                       --loss_balance y \
                       --log_to_file n \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --gpu 0 1 2 3 4 5 6 7 \
                       --data_dir ${DATA_DIR} \
                       --loss_type ${LOSS_TYPE} \
                       --max_iters ${MAX_ITERS} \
                       --checkpoints_name ${CHECKPOINTS_NAME} 
                       --distributed \
                       --optim_method ${OPTIM_METHOD} \
                       --lr_policy ${LR_POLICY} \
                       --group_method ${GROUP_METHOD} \
                       --resume_continue y \
                       --resume ./checkpoints/ade20k/${CHECKPOINTS_NAME}_latest.pth \
                        2>&1 | tee -a ${LOG_FILE}


elif [ "$1"x == "debug"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS} \
                       --phase debug --gpu 0 --log_to_file n 2>&1 | tee ${LOG_FILE}

elif [ "$1"x == "test"x ]; then
  ${PYTHON} -u main.py --configs ${CONFIGS_TEST} \
                       --data_dir ${DATA_DIR} \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test \
                       --gpu 0 1 2 3 4 5 6 7 \
                       --resume ./checkpoints/ade20k/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir ${DATA_DIR}/val/image \
                       --log_to_file n \
                       --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms

  cd lib/metrics
  ${PYTHON} -u ade20k_evaluator.py --configs ../../${CONFIGS_TEST} \
                                   --pred_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms/label \
                                   --gt_dir ${DATA_DIR}/val/label

else
  echo "$1"x" is invalid..."
fi
