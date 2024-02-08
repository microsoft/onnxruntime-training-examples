# Phi-2 Model Fine-tuning Demo

This demo will show how to use ACPT (Azure Container for PyTorch) along with accelerators such as onnxruntime training (through ORTModule) and DeepSpeed to fine-tune Phi-2 model.

## Background

[Phi-2](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)  is 2.7 billion-parameter language model with nex-t word prediction objective. It has been trained using mixture of Synthetic and Web datasets.

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

We observe **~14% speedup** for Phi-2 trained leveraging ONNX Runtime Training with 8 V100 GPUs with 32GB memory.

### Run directly on your compute

If you are using CLI by directly logging into your machine then you can follow the below instructions. The below steps assume you have the required packages like Pytorch, ONNX Runtime training, Transformers and more already installed in your system. For easier setup, you can look at the environment folder.

```bash
cd finetune-clm

# To run the model using Pytorch
torchrun --nproc_per_node 8 run_clm.py --model_name_or_path microsoft/phi-2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --save_strategy 'no' --fp16 --block_size 512 --max_steps -1 --gradient_accumulation_steps 4 --per_device_train_batch_size 2 --num_train_epochs 2 --output_dir output_dir --overwrite_output_dir --deepspeed zero_stage_2.json

# To run the model using ONNX Runtime training, you need to export couple of variables and run the same command above, overall these would be your steps:
export APPLY_ORT="True"
export ORTMODULE_FALLBACK_POLICY="FALLBACK_DISABLE"
# Optionally you can enable Triton for even faster performance
# export ORTMODULE_USE_TRITON=1
torchrun --nproc_per_node 8 run_clm.py --model_name_or_path microsoft/phi-2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --save_strategy 'no' --fp16 --block_size 512 --max_steps -1 --gradient_accumulation_steps 4 --per_device_train_batch_size 2 --num_train_epochs 2 --output_dir output_dir --overwrite_output_dir --deepspeed zero_stage_2.json
```

