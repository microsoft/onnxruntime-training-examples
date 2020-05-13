# Instructions:

Step 0. Clone this repository and switch to this BERT example directory. 

```
git clone https://github.com/microsoft/onnxruntime-training-examples.git
cd onnxruntime-training-examples/nvidia-bert
```

Step 1. Setup the BERT project workspace.

```
./setup_workspace.sh
```

This downloads the NVIDIA PyTorch BERT example and adds in files to use onnxruntime as backend.

Step 2. Build onnxruntime into Docker image.
```
cd docker
docker build --network=host -t bert-onnxruntime .
```
This builds onnxruntime from source and contains CUDA 10.1, MPI, and PyTorch 1.5.

Step 3. Download and parse training data into HDF5 format. For details follow [Getting the data](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#getting-the-data) section by NVIDIA. 

```
cd workspace
bash scripts/docker/build.sh
bash scripts/docker/launch.sh
bash data/create_datasets_from_start.sh 
```

## For local run, proceed as ..

Step 4. Set correct paths to training data for docker image.

Within workspace/scripts/docker/launch_ort.sh:
```
...
-v <replace-with-path-to-phase1-hdf5-training-data>:/data/128 
-v <replace-with-path-to-phase2-hdf5-training-data>:/data/512
...
```
Step 5. Launch interactive container.
```
cd workspace
bash scripts/docker/launch_ort.sh
```

Step 6. Modify default training parameters as needed.

Edit scripts/run_pretraining_ort.sh
```
seed=${12:-42}
num_gpus=${4:-4}

train_batch_size=${1:-8192} 
learning_rate=${2:-"6e-3"}
warmup_proportion=${5:-"0.2843"}
train_steps=${6:-7038}
accumulate_gradients=${10:-"true"}
gradient_accumulation_steps=${11:-128}

train_batch_size_phase2=${17:-4096}
learning_rate_phase2=${18:-"4e-3"}
warmup_proportion_phase2=${19:-"0.128"}
train_steps_phase2=${20:-1563}
gradient_accumulation_steps_phase2=${11:-256} 
```

Consult [Parameters](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#parameters) section by NVIDIA for additional details.

Step 7. Launch pretraining run.    
```
bash scripts/docker/launch_ort.sh
```

## For Azure run, proceed as ..

Step 4. Install Azure Cli and Azure ML CLI and SDK

```
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
az login
az extension add -n azure-cli-ml
pip install --upgrade azureml-sdk
pip install azureml-sdk[notebooks]
```
Consult [install-azure-cli](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest) for details.

Step 5. Create Azure machine learning workspace.
```
az group create --name <resource-group-name> --location <location>
az ml workspace create -w <workspace-name> -g <resource-group-name>
```
Consult [azure-ml-py](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py) or [how-to-manage-workspace-cli](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace-cli) for details.

Step 5. Create Azure container registry and upload onnxruntime Docker image.

```
az acr create --name <acr-name> --resource-group <resource-group-name> --sku <sku-type>
az acr login --name <acr-name>
docker tag bert-onnxruntime <acr-name>.azurecr.io/bert-onnxruntime
docker push <acr-name>.azurecr.io/bert-onnxruntime
```

Consult [container-registry-get-started-docker-cli](https://docs.microsoft.com/en-us/azure/container-registry/container-registry-get-started-docker-cli) for details.

Step 6. Create storage and upload training data.
    
```
az storage account create --resource-group <my-resource-group> --name <storage-name>
az storage container create --account-name <storage-name> --name <container-name>
az storage blob upload-batch --account-name <storage-name>  -d <container-name>  -s <path-to-training-data>
```
Consult [storage-account-create](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-create?tabs=azure-portal)
and [az-storage-blob-upload-batch](https://docs.microsoft.com/en-us/cli/azure/storage/blob?view=azure-cli-latest#az-storage-blob-upload-batch) for details.

Step 7. Follow further instructions in Python notebook [azureml-notebooks/run-pretraining.ipynb](azureml-notebooks/run-pretraining.ipynb)
    