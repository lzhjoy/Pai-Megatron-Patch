#! /bin/bash
START_TIME=$SECONDS

CURRENT_DIR="/opt/aps/workdir/input/file/pretrain-linear-moe/lvzhihao/megatron-for-agent/Pai-Megatron-Patch/toolkits/sft_data_preprocessing"
cd $CURRENT_DIR
MEGATRON_PATCH_PATH=$(dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATCH_PATH}:"/opt/aps/workdir/input/file/pretrain-linear-moe/lvzhihao/megatron-for-agent/YuLan-Pretrain"



input_data_path=$1
tokenizer=$2
seq_len=$3
output_data_path=$4
load_dir=$5
default_packing=$6

if [ -z ${default_packing} ]; then
  default_packing=false
fi

if [ $default_packing = true ]; then
  packing_option="\
    --sequence-packing 
  "
else
  packing_option=""
fi

cmd="python build_idxmap_sft_dataset.py \
  --input ${input_data_path} \
  --output-prefix ${output_data_path} \
  --patch-tokenizer-type ${tokenizer} \
  --load ${load_dir} \
  --seq-length ${seq_len} \
  --workers 8 \
  --partitions 1 ${packing_option}"

echo $cmd
eval $cmd

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
