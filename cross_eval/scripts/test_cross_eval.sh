#!/bin/bash
# Stage 3 smoke test: cross-architecture evaluation on quantized synthetic data.
# Verifies cross_evaluate.py: load images/labels -> train each arch -> report acc.
# Uses the post-quantization output of quantize/scripts/test_quantize.sh.
# Runs in cross_eval/ (shares core/).

cd "$(dirname "$0")/.."

# ============ Test config (small) ============
CUDA_ID=4
DATASET="ImageNet"
SUBSET="imagenette"
DATA_PATH="/data1/home/ypliu/DSproject/data"

# Stage 2 output (post-quantization). MSE_ERR=5e-7 in test_quantize.sh
IMAGES_PATH="./results/quantize_test/ImageNet/imagenette/10/smoketest/images_5e-07.pt"
LABELS_PATH="./results/quantize_test/ImageNet/imagenette/10/smoketest/labels_5e-07.pt"

# Small for smoke test: all 4 README architectures, 1 eval run, 2 training epochs
MODELS="ResNet18ImageNet,VGG11,AlexNet,ViT"
NUM_EVAL=1
EPOCH_EVAL=2
BATCH_TRAIN=256
BATCH_REAL=256

SAVE_DIR="./results/cross_eval_test/${SUBSET}"

# ============ Run ============
export CUDA_VISIBLE_DEVICES=${CUDA_ID}

mkdir -p "${SAVE_DIR}"
LOG_FILE="${SAVE_DIR}/smoketest.log"

python -u cross_evaluate.py \
    --images_path ${IMAGES_PATH} \
    --labels_path ${LABELS_PATH} \
    --dataset ${DATASET} \
    --subset ${SUBSET} \
    --data_path ${DATA_PATH} \
    --models ${MODELS} \
    --num_eval ${NUM_EVAL} \
    --epoch_eval_train ${EPOCH_EVAL} \
    --batch_train ${BATCH_TRAIN} \
    --batch_real ${BATCH_REAL} \
    --save_dir ${SAVE_DIR} \
    2>&1 | tee ${LOG_FILE}
