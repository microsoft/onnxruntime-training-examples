# Mistral 7B v0.1 Model Fine-tuning Demo

This demo will show how to use ACPT (Azure Container for PyTorch) along with accelerators such as onnxruntime training (through ORTModule) and DeepSpeed to fine-tune Mistral 7B model.

## Background

[Mistral 7B v0.1](https://mistral.ai/news/announcing-mistral-7b/) Large Language Model (LLM) is a pre-trained generative text model with 7 billion parameters.

## Set up

### AzureML
The easiest option to run the demo will be using AzureML as the environment details are already included, there is another option to run directly on the machine which is provided later. For AzureML, please complete the following prerequisites:

#### Local environment
Set up your local environment with az-cli and azureml dependency for script submission:

```
az-cli && az login
pip install azure-ai-ml azure-identity
```

#### AzureML Workspace
- An AzureML workspace is required to run this demo. Download the config.json file ([How to get config.json file from Azure Portal](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment#workspace)) for your workspace. Make sure to put this config file in this folder and name it ws_config.json.
- The workspace should have a gpu cluster. This demo was tested with GPU cluster of SKU [Standard_ND40rs_v2](https://docs.microsoft.com/en-us/azure/virtual-machines/ndv2-series). See this document for [creating gpu cluster](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=python). We do not recommend running this demo on `NC` series VMs which uses old architecture (K80).
- Additionally, you'll need to create a [Custom Curated Environment ACPT](https://learn.microsoft.com/en-us/azure/machine-learning/resource-curated-environments) with PyTorch >=2.0.1 and the requirements file in the environment folder.

## Run Experiments
The demo is ready to be run.

#### `aml_submit.py` submits an training job to AML for both Pytorch+DeepSpeed+LoRA and ORT+DeepSpeed+LoRA. This job builds the training environment and runs the fine-tuning script in it.

```bash
python aml_submit.py
```

The above script will generate two URLs, one for Pytorch and another for ONNX Runtime training.

We observe **~14% speedup** for Mistral trained leveraging ONNX Runtime Training with 8 V100 GPUs with 32GB memory.

### Run directly on your compute

If you are using CLI by directly logging into your machine then you can follow the below instructions. The below steps assume you have the required packages like Pytorch, ONNX Runtime training, Transformers and more already installed in your system. For easier setup, you can look at the environment folder.

```bash
cd finetune-clm

# To run the model using Pytorch
torchrun --nproc_per_node 8 run_clm.py --model_name_or_path mistralai/Mistral-7B-v0.1 --dataset_name databricks/databricks-dolly-15k --per_device_train_batch_size 1 --do_train --num_train_epochs 5 --output_dir results --overwrite_output_dir --save_strategy 'no' --fp16 --max_steps 500 --gradient_accumulation_steps 1 --learning_rate 0.00001 --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8 --deepspeed zero_stage_2.json

# To run the model using ONNX Runtime training, you need to export couple of variables and run the same command above, overall these would be your steps:
export APPLY_ORT="True"
export ORTMODULE_FALLBACK_POLICY="FALLBACK_DISABLE"
torchrun --nproc_per_node 8 run_clm.py --model_name_or_path mistralai/Mistral-7B-v0.1 --dataset_name databricks/databricks-dolly-15k --per_device_train_batch_size 1 --do_train --num_train_epochs 5 --output_dir results --overwrite_output_dir --save_strategy 'no' --fp16 --max_steps 500 --gradient_accumulation_steps 1 --learning_rate 0.00001 --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8 --deepspeed zero_stage_2.json
```

