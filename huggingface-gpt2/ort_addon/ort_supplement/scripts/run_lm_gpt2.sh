#!/bin/bash
set -e

num_gpus=4
use_ort=true

export TRAIN_FILE=/data/wiki.train.tokens
export TEST_FILE=/data/wiki.test.tokens

RUN_FILE=/workspace/transformers/examples/run_language_modeling_ort.py
RESULT_DIR=/workspace/results

if [ "$use_ort" = true ] ; then
    echo "Launching ORT run:"
    RUN_CMD="mpirun -n ${num_gpus} --allow-run-as-root python $RUN_FILE --ort_trainer --output_dir=$RESULT_DIR/output-ort"
else
    echo "Launching PyTorch run:"
    RUN_CMD="mpirun -n ${num_gpus} --allow-run-as-root python $RUN_FILE --output_dir=$RESULT_DIR/output-pytorch"
fi

$RUN_CMD \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --tokenizer_name=gpt2  \
    --config_name=gpt2  \
    --do_eval \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --eval_data_file=$TEST_FILE \
    --per_gpu_train_batch_size=1  \
    --per_gpu_eval_batch_size=4  \
    --gradient_accumulation_steps=16 \
    --block_size=1024  \
    --weight_decay=0.01 \
    --overwrite_output_dir \
    --logging_steps=100 \
    --num_train_epochs=5 \
