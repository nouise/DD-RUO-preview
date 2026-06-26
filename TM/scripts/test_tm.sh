#!/bin/bash
# Minimal smoke test for TM pipeline (verify stage-0 buffer + stage-1 train chain).
# Tiny num_experts / train_epochs / Iteration on a single idle GPU. Not for accuracy.
# Runs two steps: (1) buffer.py to produce a tiny expert trajectory buffer,
# (2) pool_tm.py to train the synthetic pool from that buffer.

cd "$(dirname "$0")/.."

# ============ Shared config ============
cuda_id=5
dst="ImageNet"
subset="imagenette"
net="ConvNetD5"
data_path="/data1/home/ypliu/DSproject/data"
res=128
arm=32
dim=4
layers_v="v6"
ipc=10
buffer_path="./results/tm_buffer_test"

# ============ Step 0: expert trajectory buffer (tiny) ============
export CUDA_VISIBLE_DEVICES=${cuda_id}

echo "========== Step 0: buffer =========="
python -u buffer.py \
    --dataset ${dst} \
    --subset ${subset} \
    --model ${net} \
    --res ${res} \
    --data_path ${data_path} \
    --buffer_path ${buffer_path} \
    --train_epochs 2 \
    --num_experts 10 \
    --save_interval 5 \
    --batch_train 128 \
    --dsa False \
    2>&1 | tee ${buffer_path}/buffer_smoke.log

if [ ! -d "${buffer_path}/${dst}/${subset}/${res}/${net}" ] || [ -z "$(ls -A ${buffer_path}/${dst}/${subset}/${res}/${net} 2>/dev/null)" ]; then
    echo "ERROR: buffer step produced no replay_buffer_*.pt — aborting."
    exit 1
fi
echo "Buffer produced: $(ls ${buffer_path}/${dst}/${subset}/${res}/${net}/)"

# ============ Step 1: pool_tm training (tiny) ============
sh_file="test_tm.sh"
eval_mode="S"
num_eval=1
Iteration=5
batch_syn=0
ldb=0.1
lr_img=0.001
lr_it=1000
ldb_it=10
zca=False
epoch_eval_train=2
pool_init="init"
save_path="./results/tm_test/${dst}/${subset}/${ipc}"
TAG="smoketest"
FLAG="${dst}_${subset}_${ipc}ipc_${net}_TM_${TAG}"
log_file="${save_path}/${FLAG}/smoketest.log"
mkdir -p "${save_path}/${FLAG}/"

echo "========== Step 1: pool_tm =========="
python -u pool_tm.py \
--dataset ${dst} --subset ${subset} --res ${res} \
--model ${net} \
--ipc ${ipc} \
--sh_file ${sh_file} \
--eval_mode ${eval_mode} \
--data_path ${data_path} --save_path ${save_path} --buffer_path ${buffer_path} --pool_path ${pool_init} \
--num_eval ${num_eval} \
--Iteration ${Iteration} \
--batch_syn ${batch_syn} \
--layers_v ${layers_v} \
--arm ${arm} \
--dim ${dim} \
--ldb ${ldb} --lr_img ${lr_img} --lr_it ${lr_it} --ldb_it ${ldb_it} \
--zca ${zca} \
--epoch_eval_train ${epoch_eval_train} \
--FLAG ${FLAG} 2>&1 | tee ${log_file}
