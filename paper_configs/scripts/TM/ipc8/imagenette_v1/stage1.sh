cd ..
cuda_id=0,1
dst="ImageNet"
#dst="CIFAR10"
subset="imagenette"
#subset="imagefruit"
net="ConvNetD5"
ipc=8
sh_file="run_pool_tm.sh"
eval_mode="S"
data_path="/data2/home/ypliu/DSproject/data"
save_path="/data2/home/ypliu/DSproject/results_0119"
buffer_path="/data2/home/ypliu/DSproject/buffers"

num_eval=5
Iteration=25000
batch_syn=80       # 0 means no sampling (use entire synthetic dataset)
ldb=0.1
lr_img=0.001
TAG="new0119_slice200_layersv4_con"
zca=False
lr_it=1000
res=128
ldb_it=100
FLAG="${dst}_${subset}_${ipc}ipc_${net}_TM_pool_1_${Iteration}_${ldb}_${lr_img}_${lr_it}_${ldb_it}_${res}_zca_${zca}_#${TAG}"
# Get current timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${save_path}/${FLAG}_${timestamp}_${batch_syn}.log"
echo ${log_file}

export CUDA_VISIBLE_DEVICES=${cuda_id}
nohup python -u pool_tm.py \
--dataset ${dst} --subset ${subset} --res ${res} \
--model ${net} \
--ipc ${ipc} \
--sh_file ${sh_file} \
--eval_mode ${eval_mode} \
--data_path ${data_path} --save_path ${save_path} --buffer_path ${buffer_path} \
--num_eval ${num_eval} \
--Iteration ${Iteration} \
--batch_syn ${batch_syn} \
--ldb ${ldb} --lr_img ${lr_img} --lr_it ${lr_it} --ldb_it ${ldb_it} \
--zca ${zca} \
--FLAG ${FLAG} > ${log_file} 2>&1 &
echo "Log file: ${log_file}"
echo "Process ID: $!"
