# Stable Diffusion Fine-Tuning with ACPT and ONNX Runtime

This codebase shows how to use ONNX Runtime Training along with Azure Container for PyTorch (ACPT) to fine-tune a UNet Stable Diffusion Model (from Hugging Face's diffusers library) for a text to image task.

## Run Experiments

### Run Locally
#### `StableDiffusion-finetune/finetune-code` contains all the code that is required for local testing
Relevant Files:
- finetune-code/train_text_to_image.py: fine-tuning script that leverages ONNX Runtime and DeepSpeed
- accelerate_config.py: configuration file for Hugging Face Accelerate to train on a 8 GPU machine ([Hugging Face's Accelerate](https://huggingface.co/docs/accelerate/main/en/basic_tutorials/launch) documentation)

```Dockerfile
FROM ptebic.azurecr.io/internal/azureml/aifx/nightly-ubuntu2004-cu117-py38-torch210dev:latest

RUN pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu118

# text-to-image dependencies taken from: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/requirements.txt
RUN pip install accelerate>=0.16.0 transformers>=4.25.1 datasets ftfy tensorboard Jinja2
RUN pip install git+https://github.com/huggingface/diffusers

RUN pip install azureml-core

WORKDIR workspace
RUN git clone https://github.com/microsoft/onnxruntime-training-examples.git
RUN cd onnxruntime-training-examples/StableDiffusion-finetune/finetune-code && \
        accelerate launch --config_file=accelerate_config.yaml --mixed_precision=fp16 \
        train_text_to_image.py --ort \
        --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4 \
        --dataset_name=lambdalabs/pokemon-blip-captions \
        --use_ema \
        --resolution=512 --center_crop --random_flip \
        --train_batch_size=1 \
        --gradient_accumulation_steps=4 \
        --gradient_checkpointing \
        --max_train_steps={max_train_steps} \
        --learning_rate=1e-05 \
        --max_grad_norm=1 \
        --lr_scheduler=constant --lr_warmup_steps=0 \
        --output_dir=sd-pokemon-model
```

### Run on AzureML
Set up your local environment with az-cli and azureml dependency for script submission:

```
az-cli && az login
pip install azure-ai-ml azure-identity
```

#### AzureML Workspace
- An AzureML workspace is required to run this demo. Download the config.json file ([How to get config.json file from Azure Portal](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment#workspace)) for your workspace.
- The workspace should have a gpu cluster. This demo was tested with GPU cluster of SKU [Standard_ND40rs_v2](https://docs.microsoft.com/en-us/azure/virtual-machines/ndv2-series). See this document for [creating gpu cluster](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=python).
- The submission script expects the following information in `ws_config.json` (see `ws_config_template.json` for an example):
```json
{
    "subscription_id": "subscription_id",
    "resource_group": "resource_group",
    "workspace_name": "workspace_name",
    "compute": "compute",
    "nproc_per_node": <num_GPUs>
}  
```

#### `aml_submit.py` submits an training job to AML. This job builds the training environment and runs the fine-tuning script in it.
Relevant Files:
- finetune-code/train_text_to_image.py: fine-tuning script that leverages ONNX Runtime and DeepSpeed
- accelerate_config.py: configuration file for Hugging Face Accelerate to train on a 8 GPU machine ([Hugging Face's Accelerate](https://huggingface.co/docs/accelerate/main/en/basic_tutorials/launch) documentation)
- aml_submit.py: submission script to submit training workload to AzureML
- aml_upload_util.py: util script to upload results of finetuning script to AzureML training job

Example to submit training job for CompVis/stable-diffusion-v1-4 on the lambdalabs/pokemon-blip-captions dataset:
```bash
python aml_submit.py
```

## FAQ
### Problem with Azure Authentication
If there's an Azure authentication issue, install Azure CLI [here](https://docs.microsoft.com/en-us/cli/azure/) and run `az login --use-device-code`
<br>Additionally, you can try replacing AzureCliCredential() in aml_submit.py with DefaultAzureCredential()
<br>You can learn more about Azure Identity authentication [here](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity?view=azure-python)
