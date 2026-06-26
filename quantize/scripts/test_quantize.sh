#!/bin/bash
# Stage 2 smoke test: post-quantization on a DM-produced pool checkpoint.
# Verifies quantize_pool.py: load pool -> quantize_net -> evaluate -> save images/labels.
# Runs in quantize/ (shares core/).

cd "$(dirname "$0")/.."

# ============ Test config (small) ============
CUDA_ID=4
DATASET="ImageNet"
SUBSET="imagenette"
IPC=10
DATA_PATH="/data1/home/ypliu/DSproject/data"

# Must match the DM checkpoint architecture
LAYERS_V="v6"
ARM=32
DIM=4

# Evaluation (small for smoke test)
NUM_EVAL=1
EPOCH_EVAL=2
SYN_LR=0.01
MSE_ERR=0.0000005

# DM checkpoint produced by test_dm.sh
POOL_PATH="/data1/home/ypliu/DD-RUO/DM/results/dm_test/ImageNet/imagenette/10/ImageNet_imagenette_10ipc_ConvNetD5_DM_smoketest/DM_ImageNet_ConvNetD5_10ipc_exp0_10.pt"

SAVE_DIR="./results/quantize_test/${DATASET}/${SUBSET}/${IPC}"

# ============ Run ============
export CUDA_VISIBLE_DEVICES=${CUDA_ID}

FLAG="smoketest"
LOG_DIR="${SAVE_DIR}/${FLAG}"
LOG_FILE="${LOG_DIR}/smoketest.log"
mkdir -p "${LOG_DIR}"

python -u quantize_pool.py \
    --dataset ${DATASET} \
    --subset ${SUBSET} \
    --data_path ${DATA_PATH} \
    --ipc ${IPC} \
    --model ConvNetD5 \
    --eval_mode S \
    --num_eval ${NUM_EVAL} \
    --epoch_eval_train ${EPOCH_EVAL} \
    --syn_lr_set ${SYN_LR} \
    --dsa False \
    --layers_v ${LAYERS_V} \
    --arm ${ARM} \
    --dim ${DIM} \
    --mse_err ${MSE_ERR} \
    --pool_path ${POOL_PATH} \
    --save_path ${SAVE_DIR} \
    --FLAG ${FLAG} \
    --img_version v1 \
    2>&1 | tee ${LOG_FILE}
