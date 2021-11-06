LANG=java                       # set python for py150
data_prefix=/scratch1/08401/ywen/data/ev1pt0
LITFILE=../dataset/javaCorpus/literals.json
OUTPUTDIR=save/gptneo-1.3B
PRETRAINDIR=EleutherAI/gpt-neo-1.3B
MODELTYPE=gpt-neo
LOGFILE=${OUTPUTDIR}/c2c.log
PER_NODE_GPU=4
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$PER_NODE_GPU run_gptneo.py \
        --train_filename ${data_prefix}/train/context_data.final,${data_prefix}/train/body_data.final \
        --dev_filename ${data_prefix}/test/context_data.final,${data_prefix}/test/body_data.final \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=$MODELTYPE \
        --block_size=1024 \
        --warmup_steps=1000 \
        --do_train \
        --do_eval \
        --learning_rate=5e-5 \
        --weight_decay=1e-4 \
        --per_gpu_train_batch_size=1 \
        --per_gpu_eval_batch_size=1 \
        --gradient_accumulation_steps=8 \
        --num_train_epochs=4 \
        --logging_steps=50 \
        --save_steps=500 \
        --eval_steps=500 \
        --report=wandb \
        --seed=666 \
        --deepspeed=deepspeed_config.json \
        --fp16 