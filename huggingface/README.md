# ORTModule Examples
This example uses ORTModule to fine-tune several popular [HuggingFace](https://huggingface.co/) models.

## Prerequisite
1. AzureML subscription is required to run this example. Either config.json file ([How to get config.json file from Azure Portal](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment#workspace)) or subscription_id, resource_group, workspace_name is required.
2. The subscription should have a gpu cluster. This example was tested with GPU cluster of SKU [`Standard_ND40rs_v2`](https://docs.microsoft.com/en-us/azure/virtual-machines/ndv2-series).

## Setup
1. Clone this repo
```bash
git clone https://github.com/microsoft/onnxruntime-training-examples.git
git submodule update --init --recursive
git submodule foreach git pull origin master
```
2. Install azureml-core
```bash
pip install azureml-core
```
3. Run this recipe
Sample script

```bash
cd onnxruntime-training-examples/huggingface/azureml
python hf-ort.py --workspace_name <your_workspace_name> --resource_group 
<resource_group> --subscription_id <your_subscription_id> --gpu_cluster_name V100 --min_cluster_node 0 --max_cluster_node 7 --hf_model bert-large --run_config ort
```
- This script takes ~20 mins to run. Most time is spent on building a new docker image. The step to build docker image (`hf_ort_env.register(ws).build(ws).wait_for_completion()`) can be skipped if not running for the first time.
- Choices for --hf_model are `bert-large`, `distilbert-base`, `gpt2`, `bart-large`, `t5-large`.
- Choices for --run_config ort are `pt-fp16` for PyTorch fp16, `ort` for fp16 with ORTModule, `ds_s0` for deepspeed stage 0 with pytorch, `ds_s0_ort` for deepspeed stage 0 with ORTModule, `ds_s1` for deepspeed stage 1 with pytorch, `ds_s1_ort` for deepspeed stage 1 with ORTModule.

## Common Errors
### Problem with Azure authentication
If there's an Azure authentication issue, install Azure CLI [here](https://docs.microsoft.com/en-us/cli/azure/) and run `az login --use-device-code`