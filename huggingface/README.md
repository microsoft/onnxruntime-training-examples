# What is ONNX Runtime for PyTorch

ONNX Runtime for PyTorch gives you the ability to accelerate training of large transformer PyTorch models. The training time and cost are reduced with just a one line code change.

- One line code change: ORT provides a one-line addition for existing PyTorch training scripts allowing easier experimentation and greater agility.
```python
    from torch_ort import ORTModule
    model = ORTModule(model)
```

- Flexible and extensible hardware support: The same model and API works with NVIDIA and AMD GPUs; the extensible "execution provider" architecture allow you to plug-in custom operators, optimizer and hardware accelerators.

- Faster Training: Optimized kernels provide up to 1.4X speed up in training time.

- Larger Models: Memory optimizations allow fitting a larger model such as GPT-2 on 16GB GPU, which runs out of memory with stock PyTorch.

- Composable with other acceleration libraries such as Deepspeed, Fairscale, Megatron for even faster and more efficient training

- Part of the PyTorch Ecosystem. It is available via the torch-ort python package.
 
- Built on top of highly successful and proven technologies of ONNX Runtime and ONNX format.

## Please also see
- Official ORT documentation: https://www.onnxruntime.ai/  
- Official ORT GitHub Repo: https://github.com/microsoft/onnxruntime
- Official ORT Samples Repo: https://github.com/microsoft/onnxruntime-training-examples

# ORTModule Examples
This example uses ORTModule to fine-tune several popular [HuggingFace](https://huggingface.co/) models.

## Prerequisites
1. AzureML subscription is required to run this example. Either a config.json file ([How to get config.json file from Azure Portal](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment#workspace)) or subscription_id, resource_group, workspace_name is required.
2. The subscription should have a gpu cluster. This example was tested with GPU cluster of SKU [`Standard_ND40rs_v2`](https://docs.microsoft.com/en-us/azure/virtual-machines/ndv2-series). See this document for [creating gpu cluster](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=python).

## Setup
1. Clone this repo
```bash
git clone https://github.com/microsoft/onnxruntime-training-examples.git
cd onnxruntime-training-examples
git submodule update --init --recursive
git submodule foreach git pull origin master
```
2. Make sure python 3.6+ is installed
3. Install azureml-core
```bash
pip install azureml-core
```
4. Run this recipe

### If config.json is in `huggingface/azureml`
#### BERT
```bash
cd huggingface/azureml
python hf-ort.py --gpu_cluster_name <gpu_cluster_name> --hf_model bert-large --run_config ort
```
Please see [BERT.md](BERT.md) for more details on performance gain and convergence.
### Alternatively, pass AzureML Workspace info through parameters
```bash
cd huggingface/azureml
python hf-ort.py --workspace_name <your_workspace_name> --resource_group 
<resource_group> --subscription_id <your_subscription_id> --gpu_cluster_name <gpu_cluster_name> --hf_model bert-large --run_config ort
```
- Our benchmarking and performance ran on ND40rs_v2 machine, Cuda 11, with stable release `onnxruntime_training-1.8.0%2Bcu111-cp36-cp36m-manylinux2014_x86_64.whl` from [here](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_stable_cu111.html)
- The finetuning script runs for 5-10 mins, on more available [NC24 machines](https://azure.microsoft.com/en-us/pricing/details/machine-learning/), each run will cost ~$0.3-$0.6 and will require a smaller batch size. Plus Azure container registry and storage cost.
- This script takes ~20 mins to run. Most time is spent on building a new docker image. The step to build docker image (`hf_ort_env.register(ws).build(ws).wait_for_completion()`) can be skipped if not running for the first time.
- Choices for --hf_model are `bert-large`, `distilbert-base`, `gpt2`, `bart-large`, `t5-large`, `deberta-v2-xxlarge`, `roberta-large`.
- Choices for --run_config ort are `pt-fp16` for PyTorch fp16, `ort` for fp16 with ORTModule, `ds_s0` for deepspeed stage 0 with pytorch, `ds_s0_ort` for deepspeed stage 0 with ORTModule, `ds_s1` for deepspeed stage 1 with pytorch, `ds_s1_ort` for deepspeed stage 1 with ORTModule.

## FAQ
### Problem with Azure Authentication
If there's an Azure authentication issue, install Azure CLI [here](https://docs.microsoft.com/en-us/cli/azure/) and run `az login --use-device-code`
