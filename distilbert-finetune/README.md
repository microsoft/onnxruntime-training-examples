# DistilBERT Fine-tuning Demo

This demo will show how to use ACPT along with accelerators such as onnxruntime training (through ORTModule) and DeepSpeed to fine-tune a DistilBERT model on the SQuAD dataset from the Huggingface Datasets library.

## Background

DistilBERT is a transformers based language model that has been pre-trained on a large corpus of text data. It can be fine-tuned for task such as question-answer where it reads a context paragraph and given a question it will answer based on the context.

In this demo, we will fine-tune DistilBERT using the SQuAD dataset from the Huggingface Datasets library. We will use ACPT to create our training environment and leverage some of the training acceleration technologies it offers.

## Set up

### AzureML
The demo will be run on AzureML. Please complete the following prerequisites:

#### Local environment
Set up your local environment with az-cli and azureml dependency for script submission:

```
az-cli && az login
pip install azure-ai-ml
```

#### AzureML Workspace
- An AzureML workspace is required to run this demo. Download the config.json file ([How to get config.json file from Azure Portal](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment#workspace)) for your workspace.
- The workspace should have a gpu cluster. This demo was tested with GPU cluster of SKU [Standard_ND40rs_v2](https://docs.microsoft.com/en-us/azure/virtual-machines/ndv2-series). See this document for [creating gpu cluster](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=python). We do not recommend running this demo on `NC` series VMs which uses old architecture (K80).
- Additionally, you'll need to create a custom ACPT (Azure Container for PyTorch) Curated Environment with PyTorch >=1.12.1 and the following pip dependencies:
```
pip install azureml-core accelerate datasets transformers
```

## Run Experiments
The demo is ready to be run.

#### `aml_submit.py` submits an training job to AML. This job builds the training environment and runs the fine-tuning script in it.

```bash
python aml_submit.py --ws_config [Path to workspace config json] --compute [Name of gpu cluster] --run_config [Accelerator configuration]
```

Here are the different configs and description that `aml_submit.py` takes through `--run_config` parameter.

| Config    | Description |
|-----------|-------------|
| no_acc    | PyTorch mixed precision (Default) |
| ort       | ORTModule mixed precision |
| ds        | PyTorch + Deepspeed stage 1 |
| ds_ort    | ORTModule + Deepspeed stage 1|

An example job submission to a compute target named `v100-32gb-eus` and using ORTModule + Deepspeed:

```
python aml_submit.py --ws_config ws_config.json --compute v100-32gb-eus \
    --run_config ds_ort
```

#### `inference.py` runs inferencing on your local machine while inside an ACPT image. 

```
python inference.py
python inference.py --ort
```

## FAQ
### Problem with Azure Authentication
If there's an Azure authentication issue, install Azure CLI [here](https://docs.microsoft.com/en-us/cli/azure/) and run `az login --use-device-code`
<br>Additionally, you can try replacing DefaultAzureCredential() in aml_submit.py with AzureCliCredential()
