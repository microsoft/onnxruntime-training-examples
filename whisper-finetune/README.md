# Whisper Fine-tuning Demo

This demo will show how to use ACPT (Azure Container for PyTorch) along with accelerators such as onnxruntime training (through ORTModule) and DeepSpeed to fine-tune OpenAI's whisper model on a Hindi to English speech recognition and translation task.

## Background

[Whisper](https://huggingface.co/openai/whisper-large) is a pre-trained model for automatic speech recognition (ASR) and speech translation.

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
- Additionally, you'll need to create a [Custom Curated Environment ACPT](https://learn.microsoft.com/en-us/azure/machine-learning/resource-curated-environments) with PyTorch >=1.12.1 and the following pip dependencies:
```
pip install azureml-core accelerate datasets librosa transformers
```

## Run Experiments
The demo is ready to be run.

#### `aml_submit.py` submits an training job to AML. This job builds the training environment and runs the fine-tuning script in it.

```bash
python aml_submit.py --ws_config [Path to workspace config json] --compute [Name of gpu cluster] [ORT + DeepSpeed flag]
```

An example job submission to a compute target named `v100-32gb-eus` and using ORTModule + Deepspeed:

```bash
python aml_submit.py --ws_config ws_config.json --compute v100-32gb-eus --ort_ds
```

We observe **~15% speedup** for Whisper trained leveraging ONNX Runtime Training, and Nebula Checkpointing!
![image](https://github.com/microsoft/onnxruntime-training-examples/assets/31260940/305dc251-0ece-434c-9ae5-cb409711e300)

#### `inference.py` and `inference_ort.py` runs two inferencing scenarios on your local machine. 

The inference demo requires ORT nightly which can be installed as follows along with Hugging Face Transformers for model archiecture and Librosa for soundfile loading:
```bash
pip install onnxruntime-training --pre -f https://download.onnxruntime.ai/onnxruntime_nightly_cu116.html
python -m onnxruntime.training.ortmodule.torch_cpp_extensions.install
pip install librosa transformers
```

Note: You need to download your trained weights from your training job to run inference. Script assumes pytorch_model.bin is in the same directory as inference.py

```
python inference.py # runs baseline pytorch
python inference_ort.py # runs ORT Inference
```

We observe **~200% speedup** for Whisper leveraging ONNX Runtime Inference!

## FAQ
### Problem with Azure Authentication
If there's an Azure authentication issue, install Azure CLI [here](https://docs.microsoft.com/en-us/cli/azure/) and run `az login --use-device-code`
<br>Additionally, you can try replacing AzureCliCredential() in aml_submit.py with DefaultAzureCredential()
<br>You can learn more about Azure Identity authentication [here](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity?view=azure-python)
