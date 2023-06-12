# AzureML Foundational Vision Models Fine-Tuning Examples

This codebase shows how to use ACPT (Azure Container for PyTorch) along with accelerators such as onnxruntime training (through Hugging Face Optimum) and DeepSpeed to fine-tune foundational, computer-vision models for an image classification task. Below are benchmarks on the DeepFashion using our various accelerators:

| Model                                                                                         | Batch Size | Accelerators | train_runtime (sec)           | train_samples_per_second      |
|-----------------------------------------------------------------------------------------------|------------|--------------|-------------------------------|-------------------------------|
| [apple/mobilevit-small](https://huggingface.co/apple/mobilevit-small)                         | 117        | -            | 1306.508                      | 1065.129                      |
| [apple/mobilevit-small](https://huggingface.co/apple/mobilevit-small)                         | 221        | ORT+DS       | 1215.846 **(~7.0% speedup)**  | 1144.553 **(~7.5% speedup)**  |
| [facebook/deit-base](https://huggingface.co/facebook/deit-base-patch16-224)                   | 104        | -            | 1272.333                      | 1093.739                      |
| [facebook/deit-base](https://huggingface.co/facebook/deit-base-patch16-224)                   | 124        | ORT+DS       | 1210.240 **(~5.0% speedup)**  | 1149.854 **(~5.1% speedup)**  |
| [google/vit-base](https://huggingface.co/google/vit-base-patch16-224)                         | 117        | -            | 1287.648                      | 1080.73                       |
| [google/vit-base](https://huggingface.co/google/vit-base-patch16-224)                         | 221        | ORT+DS       | 1235.098 **(~4.0% speedup)**  | 1126.712 **(~4.3% speedup)**  |
| [microsoft/beit-base](https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k)     | 110        | -            | 1496.612                      | 929.833                       |
| [microsoft/beit-base](https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k)     | 185        | ORT+DS       | 1338.658 **(~10.6% speedup)** | 1039.549 **(~11.8% speedup)** |
| [miceosoft/swinv2-base](https://huggingface.co/microsoft/swinv2-base-patch4-window12-192-22k) | 69         | -            | 2653.239                      | 524.491                       |
| [miceosoft/swinv2-base](https://huggingface.co/microsoft/swinv2-base-patch4-window12-192-22k) | 106        | ORT+DS       | 1833.086 **(~30.9% speedup)** | 759.157 **(~44.7% speedup)**  |

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

## Run Experiments
### Cloud Run:
#### `aml_submit.py` submits an training job to AML. This job builds the training environment and runs the fine-tuning script in it.

Example to submit training job for google/vit-base using maximum possible batch size:
```bash
python aml_submit.py --model_name "google/vit-base-patch16-224" --batch_size "max"
```
For more run configurations, see `model_configs` and `get_args()` in aml_submit.py

### Local Testing:
#### `image-classification/finetune-code` contains all the code that is submitted by `aml_submit.py`
```Dockerfile
FROM ptebic.azurecr.io/public/azureml/aifx/stable-ubuntu2004-cu117-py38-torch1131:ort1.15.0-vision-patch
RUN pip install accelerate datasets evaluate optimum transformers
RUN pip install azureml-core scikit-learn

WORKDIR workspace
RUN git clone https://github.com/microsoft/onnxruntime-training-examples.git
 
# Determine the following:
#   - nproc_per_node: how many GPUs you would like to use for distributed fine-tuning
#   - model: model name (can be found by inspecting model_configs in aml_submit.py)
#   - dataset: Local (https://huggingface.co/docs/datasets/image_dataset) OR Hugging Face Hub Dataset 
#   - bs: model/accelerator-specific batch size (can be found by inspecting model_configs in aml_submit.py)
RUN cd `image-classification/finetune-code` && \
    torchrun --nproc_per_node={nproc_per_node} run_image_classification_ort.py \
             --model_name_or_path {model} \
             --do_train --do_eval \
             --train_dir {dataset}/train --validation_dir {dataset}/validation \
             --fp16 True --num_train_epochs 100 \
             --per_device_train_batch_size {bs} --per_device_eval_batch_size {bs} \
             --remove_unused_columns False --ignore_mismatched_sizes True \
             --output_dir output_dir --overwrite_output_dir --dataloader_num_workers {2*dataloader_num_workers} \
             --optim adamw_ort_fused --deepspeed zero_stage_1.json
```

## FAQ
### Problem with Azure Authentication
If there's an Azure authentication issue, install Azure CLI [here](https://docs.microsoft.com/en-us/cli/azure/) and run `az login --use-device-code`
<br>Additionally, you can try replacing AzureCliCredential() in aml_submit.py with DefaultAzureCredential()
<br>You can learn more about Azure Identity authentication [here](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity?view=azure-python)
