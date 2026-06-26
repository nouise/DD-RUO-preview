#!/bin/bash
# Minimal smoke test for DC pipeline (verify the full chain runs end-to-end).
# Tiny ipc / Iteration / num_eval on a single idle GPU. Not for accuracy.

cd "$(dirname "$0")/.."

# ============ Test config (small) ============
cuda_id=4
dst="ImageNet"
subset="imagenette"
net="ConvNetD5"
ipc=10
sh_file="test_dc.sh"
eval_mode="S"
data_path="/data1/home/ypliu/DSproject/data"
num_eval=1
Iteration=2
batch_syn=0
ldb=5
lr_img=0.001
arm=32
dim=4
layers_v="v6"
save_path="./results/dc_test/${dst}/${subset}/${ipc}"
pool_init="init"

method="DC"

TAG="smoketest"
zca=False
lr_it=100
res=128
ldb_it=1500
FLAG="${dst}_${subset}_${ipc}ipc_${net}_${method}_${TAG}"
log_file="${save_path}/${FLAG}/smoketest.log"
echo "Log file: ${log_file}"
mkdir -p "${save_path}/${FLAG}/"

export CUDA_VISIBLE_DEVICES=${cuda_id}
python -u pool_dc.py \
--dataset ${dst} --subset ${subset} --res ${res} \
--model ${net} \
--method ${method} \
--ipc ${ipc} \
--layers_v $layers_v \
--arm $arm \
--dim $dim \
--sh_file ${sh_file} \
--eval_mode ${eval_mode} \
--data_path ${data_path} --save_path ${save_path} --pool_path ${pool_init} \
--num_eval ${num_eval} \
--Iteration ${Iteration} \
--zca ${zca} \
--batch_syn ${batch_syn} \
--ldb ${ldb} --lr_img ${lr_img} --lr_it ${lr_it} --ldb_it ${ldb_it} \
--FLAG ${FLAG} 2>&1 | tee ${log_file}
