#!/bin/bash

export WANDB_PROJECT=gptneo
export UCX_TLS=knem,dc_x,rc
LOCAL_RANK=$PMI_RANK

NGPUS=1
NNODES=1
MASTER=""
MVAPICH=false

while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | sed 's/^[^=]*=//g'`
    if [[ "$VALUE" == "$PARAM" ]]; then
        shift
        VALUE=$1
    fi
    case $PARAM in
        --help)
            echo "USAGE: ./launch_node_torch_imagenet.sh"
            echo "  --help           Display this help message"
            echo "  --ngpus [count]  Number of GPUs per node (default: 1)"
            echo "  --nnodes [count] Number of nodes this script is launched on (default: 1)"
            echo "  --master [addr]  Address of master node (default: \"\")"
            echo "  --mvapich           Use MVAPICH env variables for initialization (default: false)"
            exit 0
        ;;
        --ngpus)
            NGPUS=$VALUE
        ;;
        --nnodes)
            NNODES=$VALUE
        ;;
        --master)
            MASTER=$VALUE
        ;;      
        --mvapich)
            MVAPICH=true
        ;;
        *)
          echo "ERROR: unknown parameter \"$PARAM\""
          exit 1
        ;;
    esac
    shift
done

export UCX_TLS=knem,dc_x,rc
LOCAL_RANK=$PMI_RANK

source ~/.bashrc
module unload cuda
module load cuda/11.0
export CUDA_HOME=/opt/apps/cuda/11.0
conda activate cuda11torch1.9 
which python

echo Launching torch.distributed: nproc_per_node=$NGPUS, nnodes=$NNODES, master_addr=$MASTER, local_rank=$LOCAL_RANK, using_mvapich=$MVAPICH

LANG=java                       # set python for py150
data_prefix=/scratch/ywen/data
LITFILE=../dataset/javaCorpus/literals.json
PRETRAINDIR=EleutherAI/gpt-neo-1.3B
GRAD_ACC=1
WEIGHT_DECAY=1e-4
LR=8e-5
OUTPUTDIR=save/gptneo-1.3B_${LR}_newdata_${NNODES}nodes_acc${GRAD_ACC}_wd${WEIGHT_DECAY}
LOGFILE=${OUTPUTDIR}/trainer.log
python -m torch.distributed.launch \
    --nproc_per_node=$NGPUS \
    --nnodes=$NNODES \
    --node_rank=$LOCAL_RANK \
    --master_addr=$MASTER \
  run_gptneo.py \
    --train_filename ${data_prefix}/train/context_data.final,${data_prefix}/train/body_data.final \
    --dev_filename ${data_prefix}/test/context_data.final,${data_prefix}/test/body_data.final \
    --lit_file=$LITFILE \
    --langs=$LANG \
    --output_dir=$OUTPUTDIR \
    --pretrain_dir=$PRETRAINDIR \
    --log_file=$LOGFILE \
    --gradient_accumulation_steps=$GRAD_ACC \
    --model_type=gpt-neo \
    --block_size=1024 \
    --warmup_steps=1000 \
    --do_train \
    --do_eval \
    --learning_rate=$LR \
    --weight_decay=$WEIGHT_DECAY \
    --per_gpu_train_batch_size=1 \
    --per_gpu_eval_batch_size=1 \
    --num_train_epochs=2 \
    --logging_steps=50 \
    --save_steps=1500 \
    --eval_steps=1500 \
    --report_to=wandb \
    --seed=666 \
    --deepspeed=deepspeed_config.json \
    --fp16
