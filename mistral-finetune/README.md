# Mistral 7B v0.1 Model Fine-tuning Demo

This demo will show how to use ACPT (Azure Container for PyTorch) along with accelerators such as onnxruntime training (through ORTModule) and DeepSpeed to fine-tune Mistral 7B model.

## Background

[Mistral 7B v0.1](https://mistral.ai/news/announcing-mistral-7b/) Large Language Model (LLM) is a pre-trained generative text model with 7 billion parameters.

## Set up

### AzureML
The demo will be run on AzureML. Please complete the following prerequisites:

#### Local environment
Set up your local environment with az-cli and azureml dependency for script submission:

```
az-cli && az login
pip install azure-ai-ml azure-identity
```

#### AzureML Workspace
- An AzureML workspace is required to run this demo. Download the config.json file ([How to get config.json file from Azure Portal](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment#workspace)) for your workspace.
- The workspace should have a gpu cluster. This demo was tested with GPU cluster of SKU [Standard_ND40rs_v2](https://docs.microsoft.com/en-us/azure/virtual-machines/ndv2-series). See this document for [creating gpu cluster](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=python). We do not recommend running this demo on `NC` series VMs which uses old architecture (K80).
- Additionally, you'll need to create a [Custom Curated Environment ACPT](https://learn.microsoft.com/en-us/azure/machine-learning/resource-curated-environments) with PyTorch >=2.0.1 and the following pip dependencies:
```
pip install azureml-core accelerate datasets librosa transformers
```

## Run Experiments
The demo is ready to be run.

#### `aml_submit.py` submits an training job to AML for both Pytorch+DeepSpeed+LoRA and ORT+DeepSpeed+LoRA. This job builds the training environment and runs the fine-tuning script in it.

```bash
python aml_submit.py --ws_config [Path to workspace config json]
```

We observe **~14% speedup** for Mistral trained leveraging ONNX Runtime Training!
![image](https://github.com/microsoft/onnxruntime-training-examples/assets/31260940/305dc251-0ece-434c-9ae5-cb409711e300)

