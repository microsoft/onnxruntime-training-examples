This example shows how ONNX Runtime training can be done on BERT pretraining implementation in PyTorch maintained at https://github.com/NVIDIA/DeepLearningExamples.

Steps:
  * [Prepare data](#prepare-data)
  * [Run in Azure Machine Learning service](#bert-pretraining-with-onnx-runtime-in-azure-machine-learning-service)
  * [Run in DGX-2](#bert-pretraining-with-onnx-runtime-in-dgx-2)

## Prepare data
Please refer to [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#getting-the-data) repo for detailed instructions for data preparation. The following are minimal set of instructions to download and process one of the datasets used for BERT pretraining.

Note that the datasets used for BERT pretraining are large and need lot of disk space to downloand process the data. After processing, data should be made available for training. Due to the large size of the data copy, it is recommended to the execute the steps below in the training environment itself or in an environment from where data transfer to training environment will be fast and efficient. See specific data prepration instructions for Azure ML and DGx-2 in the respective sections below.

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
mkdir -p workspace && mv DeepLearningExamples/PyTorch/LanguageModeling/BERT/ workspace/
rm -rf DeepLearningExamples
cd workspace
git clone https://github.com/attardi/wikiextractor.git
cd ..
```
### Download and process data
Download and prepare Wikicorpus training data in HDF5 format. If you want to include additional datasets referenced in [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#getting-the-data), you need to update the following instructions to include them.

__Pre-requisite__ 
* Install Natural Language Toolkit (NLTK) `python3-pip install nltk`
* Ensure the `python` is sym-linked to `python3`: `sudo ln -s /usr/bin/python3 /usr/bin/python`

```bash
export BERT_PREP_WORKING_DIR=./workspace/BERT/data/

# Download
python3 ./workspace/BERT/data/bertPrep.py --action download --dataset wikicorpus_en
python3 ./workspace/BERT/data/bertPrep.py --action download --dataset google_pretrained_weights

# Properly format the text files
# fixing path issue in code (it should have used BERT_PREP_WORKING_DIR as prefix for path instead of hardcoded prefix)
sed -i "s/path_to_wikiextractor_in_container = '/path_to_wikiextractor_in_container = './g" ./workspace/BERT/data/bertPrep.py
python3 ./workspace/BERT/data/bertPrep.py --action text_formatting --dataset wikicorpus_en

# Shard the text files
python3 ./workspace/BERT/data/bertPrep.py --action sharding --dataset wikicorpus_en

# Create HDF5 files Phase 1
python3 ./workspace/BERT/data/bertPrep.py --action create_hdf5_files --dataset wikicorpus_en --max_seq_length 128 \
 --max_predictions_per_seq 20 --vocab_file ./workspace/BERT/data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt --do_lower_case 1

# Create HDF5 files Phase 2
python3 ./workspace/BERT/data/bertPrep.py --action create_hdf5_files --dataset wikicorpus_en --max_seq_length 512 \
 --max_predictions_per_seq 80 --vocab_file ./workspace/BERT/data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt --do_lower_case 1

```

### Make data accessible for training
Data prepared using the steps above need to be available for training. Follow instructions in the sections below to learn steps required for making data accessbile to training process depending on the environment where BERT pretraining will be done.

## BERT pretraining with ONNX Runtime in Azure Machine Learning service

#### Environment for data preparation for use in Azure ML
Please refer to the [stroage guidance](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-access-data#storage-guidance) for details on using Azure storage account for training in AzureML. To use data in distributed training, the recommendation in Azure ML is to host the data in a blob container in an Azure Storage account, register that blob container as a data store and mount it in the compute targets used for training. If the data to use with pretraing is already processed and available, it should be transferred to a blob container. 

To do data preparation from scratch and use it in a non-production setup, it is easier to execute the steps in an Azure ML compute instance and directly in the path associated with mounted `workspacefilestore` (the path will look like `/mnt/batch/tasks/shared/LS_root/mounts/clusters/<compute_instance_name>`). This will help with making the processed data available in the training compute target by simply attaching `workspacefilestore`. For high performance in scenarios involving large datasets, `workspacefilestore` may not be used and a blob based datastore in standard or premium storage is recommended.

#### Pretraining

The BERT pretraining job in Azure ML can be launched using the following options:
1. Azure ML [Compute Instance](https://docs.microsoft.com/en-us/azure/machine-learning/concept-compute-instance)
2. Azure ML [CLI](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-train-deploy-model-cli) or [SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py)

For instructions to use Python SDK follow the steps in the Python notebook [azureml-notebooks/run-pretraining.ipynb](azureml-notebooks/run-pretraining.ipynb). If you have a local setup to run an Azure ML notebook, you could run the steps in the notebook in that environment. Otherwise, a compute instance in AzureML could be created and used to run the steps.

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

    
