This example shows how ONNX Runtime training can be done on BERT pretraining implementation maintained at https://github.com/NVIDIA/DeepLearningExamples.

## Preparing data
Please refer to [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#getting-the-data) repo for detailed instructions for data preparation. The following are minimal set of instructions to download and process one of the datasets used for BERT pretraining.

### Get Code
```bash
git clone --no-checkout https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/
git config core.sparseCheckout true
echo "PyTorch/LanguageModeling/BERT/*"> .git/info/sparse-checkout
git checkout 4733603577080dbd1bdcd51864f31e45d5196704
cd ..
```

### Setup working directory

```bash
mkdir -p workspace && 
    mv DeepLearningExamples/PyTorch/LanguageModeling/BERT/ workspace/
```
### Download and process data
Download and prepare Wikicorpus training data in HDF5 format. If you want to include additional datasets referenced in [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#getting-the-data), you need to update the following instructions to include them.

```bash
# Download
python3 ./workspace/bert/data/bertPrep.py --action download --dataset wikicorpus_en
python3 ./workspace/bert/data/bertPrep.py --action download --dataset google_pretrained_weights

# Properly format the text files
python3 ./workspace/bert/data/bertPrep.py --action text_formatting --dataset wikicorpus_en

# Shard the text files
python3 ./workspace/bert/data/bertPrep.py --action sharding --dataset wikicorpus_en

# Create HDF5 files Phase 1
python3 ./workspace/bert/data/bertPrep.py --action create_hdf5_files --dataset wikicorpus_en --max_seq_length 128 \
 --max_predictions_per_seq 20 --vocab_file ./workspace/bert/data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt --do_lower_case 1

# Create HDF5 files Phase 2
python3 ./workspace/bert/data/bertPrep.py --action create_hdf5_files --dataset wikicorpus_en --max_seq_length 512 \
 --max_predictions_per_seq 80 --vocab_file ./workspace/bert/data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt --do_lower_case 1

```

## BERT pretraining with ONNX Runtime in Azure Machine Learning Service
The pretraining job in Azure ML can be launched using the following options:
1. Azure ML [Compute Instance](https://docs.microsoft.com/en-us/azure/machine-learning/concept-compute-instance)
2. Azure ML [CLI](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-train-deploy-model-cli) or [SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py)

For instructions to use Python SDK follow the steps in the Python notebook [azureml-notebooks/run-pretraining.ipynb](azureml-notebooks/run-pretraining.ipynb).

## BERT pretraining with ONNX Runtime in DGX-2

Step 4. Set correct paths to training data for docker image.

Within workspace/scripts/docker/launch_ort.sh:
```bash
...
-v <replace-with-path-to-phase1-hdf5-training-data>:/data/128 
-v <replace-with-path-to-phase2-hdf5-training-data>:/data/512
...
```
Step 5. Launch interactive container.
```bash
cd workspace
bash scripts/docker/launch_ort.sh
```

Step 6. Modify default training parameters as needed.

Edit scripts/run_pretraining_ort.sh
```bash
seed=${12:-42}
num_gpus=${4:-4}
gpu_memory_limit_gb=${26:-"32"}

accumulate_gradients=${10:-"true"}
partition_optimizer=${27:-"false"}

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

Be sure to set the number of GPUs and the per GPU memory limit in GB.
The per GPU batch size will be the training batch size divided by gradient accumulation steps.
Consult [Parameters](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#parameters) section by NVIDIA for additional details.

Step 7. Launch pretraining run.    
```bash
bash scripts/run_pretraining_ort.sh
```
If you get memory errors, try reducing the batch size or enabling the partition optimizer flag.

## For Azure run, proceed as ..

Step 4. Install Azure Cli and Azure ML CLI and SDK

```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
az login
az extension add -n azure-cli-ml
pip install --upgrade azureml-sdk
pip install azureml-sdk[notebooks]
```
Consult [install-azure-cli](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest) for details.

Step 5. Create Azure machine learning workspace.
```bash
az group create --name <resource-group-name> --location <location>
az ml workspace create -w <workspace-name> -g <resource-group-name>
```
Consult [azure-ml-py](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py) or [how-to-manage-workspace-cli](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace-cli) for details.

Step 6. Create Azure container registry and upload onnxruntime Docker image.

```bash
az acr create --name <acr-name> --resource-group <resource-group-name> --sku <sku-type>
az acr login --name <acr-name>
docker tag bert-onnxruntime <acr-name>.azurecr.io/bert-onnxruntime
docker push <acr-name>.azurecr.io/bert-onnxruntime
```

Consult [container-registry-get-started-docker-cli](https://docs.microsoft.com/en-us/azure/container-registry/container-registry-get-started-docker-cli) for details.

Step 7. Create storage and upload training data.
    
```bash
az storage account create --resource-group <my-resource-group> --name <storage-name>
az storage container create --account-name <storage-name> --name <container-name>
az storage blob upload-batch --account-name <storage-name>  -d <container-name>  -s <path-to-training-data>
```
Consult [storage-account-create](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-create?tabs=azure-portal)
and [az-storage-blob-upload-batch](https://docs.microsoft.com/en-us/cli/azure/storage/blob?view=azure-cli-latest#az-storage-blob-upload-batch) for details.

Step 8. Follow further instructions in Python notebook [azureml-notebooks/run-pretraining.ipynb](azureml-notebooks/run-pretraining.ipynb)
    
