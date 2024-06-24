# Llama-3 Model Fine-tuning Demo

This demo will show how to use ACPT (Azure Container for PyTorch) along with accelerators such as onnxruntime training (through ORTModule) and DeepSpeed to fine-tune Llama-3 model.

## Background

[Llama-3-8b](https://huggingface.co/blog/llama3) the latest large language model from Meta, is built on the architecture of Llama 2 and comes in two sizes (8B and 70B parameters) with pre-trained and instruction-tuned versions. Here we show fine-tuning on 8b model.

## Set up

### AzureML
The easiest option to run the demo will be using AzureML as the environment details are already included, there is another option to run directly on the machine which is provided later. For AzureML, please complete the following prerequisites:

#### Local environment
Follow [Install Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-linux?pivots=apt#install-azure-cli) to install Azure CLI.
Set up your local environment with az-cli and azureml dependency for script submission:

```
az login
pip install azure-ai-ml azure-identity
```

#### AzureML Workspace
- An AzureML workspace is required to run this demo. Download the config.json file ([How to get config.json file from Azure Portal](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment#workspace)) for your workspace. Make sure to put this config file in this folder and name it ws_config.json.
- The workspace should have a gpu cluster. This demo was tested with GPU cluster of SKU [Standard_ND96asr_A100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/nda100-v4-series). [Standard_ND40rs_v2](https://docs.microsoft.com/en-us/azure/virtual-machines/ndv2-series) should also work with reduced block size. See this document for [creating gpu cluster](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=python). We do not recommend running this demo on `NC` series VMs which uses old architecture (K80).
- Additionally, you'll need to create a [Custom Curated Environment ACPT](https://learn.microsoft.com/en-us/azure/machine-learning/resource-curated-environments) with PyTorch >=2.2.0 and the steps in the Dockerfile.

## Run Experiments
The demo is ready to be run.

#### `aml_submit_clm.py` submits an training job to AML for both Pytorch+DeepSpeedStage2+LoRA and ORT+DeepSpeedStage2+LoRA. This job builds the training environment and runs the fine-tuning script in it.

```bash
python aml_submit_clm.py
```

The above script will generate two URLs, one for Pytorch and another for ONNX Runtime training.

We observe **12% speedup** for Llama-3 trained leveraging ONNX Runtime Training with 8 V100 GPUs with 32GB memory with batch size of 1.

### Run directly on your compute

If you are using CLI by directly logging into your machine then you can follow the below instructions. The below steps assume you have the required packages like Pytorch, ONNX Runtime training, Transformers and more already installed in your system. For easier setup, you can look at the environment folder.

```bash
cd finetune-clm

# To run the model using Pytorch
torchrun --nproc_per_node 8 run_clm.py --model_name_or_path meta-llama/Meta-Llama-3-8B --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --save_strategy 'no' --fp16 --block_size 2048 --max_steps -1 --per_device_train_batch_size 1 --num_train_epochs 2 --output_dir output_dir --overwrite_output_dir --deepspeed zero_stage_2.json --evaluation_strategy no --remove_unused_columns False

# To run the model using ONNX Runtime training, you need to export couple of variables and run the same command above, overall these would be your steps:
export APPLY_ORT="True"
export ORTMODULE_FALLBACK_POLICY="FALLBACK_DISABLE"
export ORTMODULE_DEEPCOPY_BEFORE_MODEL_EXPORT=0
# Optionally you can enable/disable Triton, for faster performance it is turned on
export ORTMODULE_USE_TRITON=1
torchrun --nproc_per_node 8 run_clm.py --model_name_or_path meta-llama/Meta-Llama-3-8B --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --save_strategy 'no' --fp16 --block_size 2048 --max_steps -1 --per_device_train_batch_size 1 --num_train_epochs 2 --output_dir output_dir --overwrite_output_dir --deepspeed zero_stage_2.json --evaluation_strategy no --remove_unused_columns False
```

