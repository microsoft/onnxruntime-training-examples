#!/bin/bash

echo '' > /logs/benchmark.txt

for config in ort pt-fp16; do 
for model in bert-large distilbert-base gpt2 bart-large t5-large deberta-v2-xxlarge roberta-large; do
for ngpu in 1 8; do

  if [ "$ngpu" == "8" ] && [ "$config" == "pt-fp16" ]; then
    continue
  fi 
       
  echo "submitting model $model for config $config on $ngpu gpus"
  command="python /stage/onnxruntime-training-examples/huggingface/azureml/hf-ort.py"
  command+=" --run_config $config --hf_model $model --gpu_cluster_name local --process_count $ngpu"

  echo $command
  $command 2>&1 | tee /logs/torch-$config-$model-$ngpu-gpu.txt
  echo "torch-$config-$model-$ngpu-gpu.txt" >> /logs/benchmark.txt
  perf=`grep stable_train_samples_per_second\' /workspace/torch-$config-$model-$ngpu-gpu.txt`
  echo $perf >> /logs/benchmark.txt
  echo "" >> /logs/benchmark.txt
  echo "" 

done; done; done
~    

