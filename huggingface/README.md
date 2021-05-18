# ORTModule Examples
This example uses ORTModule to fine-tune several popular HuggingFace models.

## Prerequisite
1. AzureML subscription. subscription_id, resource_group, workspace_name is required to the script.

## Setup
1. Clone this repo
```bash
git clone https://github.com/microsoft/onnxruntime-training-examples.git
git submodule init
git submodule update
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
