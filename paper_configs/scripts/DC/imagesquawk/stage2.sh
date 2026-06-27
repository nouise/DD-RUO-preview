cd ..
cuda_id=3
dst="ImageNet"
#dst="CIFAR10"
#subset="imagenette"
subset="imagesquawk"
net="ConvNetD5"
ipc=96
sh_file="run_pool_dc.sh"
eval_mode="S"
data_path="/data1/home/ypliu/DSproject/data"
num_eval=5
Iteration=500
batch_syn=0 # 0 means no sampling (use entire synthetic dataset)
ldb=5
lr_img=0.001
arm=32
dim=4
layers_v="v6"
TAG="layers=${layers_v}_arm=${arm}_dim=${dim}_stage2"
zca=False
lr_it=100
res=128
ldb_it=1500
save_path="/data1/home/ypliu/DSproject/dc_result/${dst}/${subset}/${ipc}"
pool_init="/data1/home/ypliu/DSproject/dc_result/ImageNet/imagesquawk/96/ImageNet_imagesquawk_96ipc_ConvNetD5_DC_DC_pool_1_1000_5_0.001_100_700_128_zca_False_#layers=v6_arm=32_dim=4_stage1/20250222_180748_ImageNet_ConvNetD5_96_250_dc_pool.pt"
method="DC"

FLAG="${dst}_${subset}_${ipc}ipc_${net}_${method}_DC_pool_1_${Iteration}_${ldb}_${lr_img}_${lr_it}_${ldb_it}_${res}_zca_${zca}_#${TAG}"
# Get current timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${save_path}/${FLAG}/${timestamp}_${batch_syn}.log"
echo ${log_file}
mkdir -p ${save_path}/${FLAG}/
export CUDA_VISIBLE_DEVICES=${cuda_id}
nohup python -u pool_dc.py \
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
--FLAG ${FLAG} > ${log_file} 2>&1 &
echo "Log file: ${log_file}"
echo "Process ID: $!"
