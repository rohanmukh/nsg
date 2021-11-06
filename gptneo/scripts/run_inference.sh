LANG=java                       # set python for py150
data_prefix=/scratch/ywen/data
LITFILE=../dataset/javaCorpus/literals.json
OUTPUTDIR=save/gptneo-1.3B_inference
PRETRAINDIR=EleutherAI/gpt-neo-1.3B
LOAD_PATH=save/gptneo-1.3B/checkpoint-40000
LOGFILE=${OUTPUTDIR}/c2c_inference.log
partition=-1
MODELTYPE=gpt-neo
CUDA_VISIBLE_DEVICES=0 python run_gptneo.py \
    --dev_filename ${data_prefix}/test/context_data.final,${data_prefix}/test/body_data.final \
    --lit_file=$LITFILE \
    --load_name=$LOAD_PATH \
    --langs=$LANG \
    --output_dir=$OUTPUTDIR \
    --pretrain_dir=$PRETRAINDIR \
    --log_file=$LOGFILE \
    --model_type=$MODELTYPE \
    --block_size=512 \
    --do_test \
    --test_partition $partition \
    --per_gpu_eval_batch_size=1 \
    --beam_size 10 \
    --seed=666
