# Accelerate BERT pre-training with ONNX Runtime

This example uses ONNX Runtime to pre-train the BERT PyTorch model maintained at https://github.com/NVIDIA/DeepLearningExamples.

You can run the training in Azure Machine Learning or on an NVIDIA DGX-2.

## Setup

1. Clone this repo

    ```bash
    git clone https://github.com/microsoft/onnxruntime-training-examples.git
    cd onnxruntime-training-examples
    ```

2. Clone download code and model

    ```bash
    git clone --no-checkout https://github.com/NVIDIA/DeepLearningExamples.git
    cd DeepLearningExamples/
    git config core.sparseCheckout true
    echo "PyTorch/LanguageModeling/BERT/*"> .git/info/sparse-checkout
    git checkout 4733603577080dbd1bdcd51864f31e45d5196704
    cd ..
    ```

3. Create working directory

    ```bash
    mkdir -p workspace
    mv DeepLearningExamples/PyTorch/LanguageModeling/BERT/ workspace
    rm -rf DeepLearningExamples
    cp -r ./nvidia-bert/ort_addon/* workspace/BERT
    cd workspace
    git clone https://github.com/attardi/wikiextractor.git
    cd ..
    ```

## Download and prepare data

The following are a minimal set of instructions to download and process one of the datasets used for BERT pre-training.

To include additional datasets, and for more details, refer to [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#getting-the-data).

Note that the datasets used for BERT pre-training need a large amount of disk space. After processing, the data should be made available for training. Due to the large size of the data copy, we recommend that you execute the steps below in the training environment itself or in an environment from where data transfer to training environment will be fast and efficient.

1. Check pre-requisites

    * Natural Language Toolkit (NLTK) `python3-pip install nltk`
    * Python 3.6

2. Download and prepare Wikicorpus training data in HDF5 format.

    ```bash
    export BERT_PREP_WORKING_DIR=./workspace/BERT/data/

    # Download
    python ./workspace/BERT/data/bertPrep.py --action download --dataset wikicorpus_en
    python ./workspace/BERT/data/bertPrep.py --action download --dataset google_pretrained_weights

    # Fix path issue to use BERT_PREP_WORKING_DIR as prefix for path instead of hard-coded prefix
    sed -i "s/path_to_wikiextractor_in_container = '/path_to_wikiextractor_in_container = './g" ./workspace/BERT/data/bertPrep.py

    # Format text files
    python ./workspace/BERT/data/bertPrep.py --action text_formatting --dataset wikicorpus_en

    # Shard text files
    python ./workspace/BERT/data/bertPrep.py --action sharding --dataset wikicorpus_en

    # Fix path to workspace to allow running outside of the docker container
    sed -i "s/python \/workspace/python .\/workspace/g" ./workspace/BERT/data/bertPrep.py

    # Create HDF5 files Phase 1
    python ./workspace/BERT/data/bertPrep.py --action create_hdf5_files --dataset wikicorpus_en --max_seq_length 128 \
      --max_predictions_per_seq 20 --vocab_file ./workspace/BERT/data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt --do_lower_case 1

    # Create HDF5 files Phase 2
    python ./workspace/BERT/data/bertPrep.py --action create_hdf5_files --dataset wikicorpus_en --max_seq_length 512 \
    --max_predictions_per_seq 80 --vocab_file ./workspace/BERT/data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt --do_lower_case 1
    ```

3. Make data accessible for training

    Data prepared using the steps above need to be available for training. Follow instructions in the sections below to learn steps required for making data accessible to training process depending on the training environment.

## BERT pre-training with ONNX Runtime in Azure Machine Learning

1. Setup environment

    * Transfer training data to Azure blob storage
    * Register the blob container as a data store
    * Mount the data store in the compute targets used for training

    Please refer to the [storage guidance](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-access-data#storage-guidance) for more details on using Azure storage account for training in Azure Machine Learning.

2. Execute pre-training

    The BERT pre-training job in Azure Machine Learning can be launched using either of these environments:

    * Azure Machine Learning [Compute Instance](https://docs.microsoft.com/en-us/azure/machine-learning/concept-compute-instance) to run the Jupyter notebook.
    * Azure Machine Learning [SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py)

    You will need a [GPU optimized compute target](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets#amlcompute) - _either NCv3 or NDv2 series_, to execute this pre-training job.

    Execute the steps in the Python notebook [azureml-notebooks/run-pretraining.ipynb](azureml-notebooks/run-pretraining.ipynb) within your environment. If you have a local setup to run an Azure ML notebook, you could run the steps in the notebook in that environment. Otherwise, a compute instance in Azure Machine Learning could be created and used to run the steps.

## BERT pre-training with ONNX Runtime on a DGX-2

1. Check pre-requisites

    * CUDA 10.1
    * Docker
    * [NVIDIA docker toolkit](https://github.com/NVIDIA/nvidia-docker)

2. Pull the ONNX Runtime training docker image

    ```bash
    docker pull mcr.microsoft.com/azureml/onnxruntime-training:0.1-rc1-openmpi4.0-cuda10.1-cudnn7.6-nccl2.4.8
    ```

3. Set correct paths to training data for docker image.

   Edit `nvida-bert/docker/launch.sh`.

   ```bash
   ...
   -v <replace-with-path-to-phase1-hdf5-training-data>:/data/128
   -v <replace-with-path-to-phase2-hdf5-training-data>:/data/512
   ...
   ```

4. Set the number of GPUs and per GPU limit.

    Edit `workspace/BERT/scripts/run_pretraining_ort.sh`.

    ```bash
    num_gpus=${4:-8}
    gpu_memory_limit_gb=${26:-"32"}
    ```

5. Modify other training parameters as needed.

    Edit `workspace/BERT/scripts/run_pretraining_ort.sh`.

    ```bash
    seed=${12:-42}

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

    The per GPU batch size will be the training batch size divided by gradient accumulation steps.

    Consult [Parameters](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#parameters) section by NVIDIA for additional details.

6. Launch interactive container.

    ```bash
    cd workspace/BERT
    bash ../../nvidia-bert/docker/launch.sh
    ```

7. Launch pre-training run

    ```bash
    bash /workspace/bert/scripts/run_pretraining_ort.sh
    ```

    If you get memory errors, try reducing the batch size or enabling the partition optimizer flag.

## Fine-tuning

For fine-tuning tasks, follow [model_evaluation.md](model_evaluation.md)
