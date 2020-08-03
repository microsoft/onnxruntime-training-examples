# Accelerate BERT pre-training with ONNX Runtime

This example uses ONNX Runtime to pre-train the BERT PyTorch model maintained at https://github.com/NVIDIA/DeepLearningExamples.

You can run the training in Azure Machine Learning or on an ND40rs\_v2.

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

    After completing the steps above, data in hdf5 format will be available at the following locations: 

    * Phase 1 data: `./workspace/BERT/data/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en/train`
    * Phase 2 data: `./workspace/BERT/data/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en/train`

    Below instructions refer to these hdf5 data files as the data to make accessible to training process.

## BERT pre-training with ONNX Runtime in Azure Machine Learning

1. Data Transfer

    * Transfer training data to Azure blob storage

    To transfer the data to an Azure blob storage using [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest), use command:
    ```bash
    az storage blob upload-batch --account-name <storage-name> -d <container-name> -s ./workspace/BERT/data
    ```

    * Register the blob container as a data store
    * Mount the data store in the compute targets used for training

    Please refer to the [storage guidance](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-access-data#storage-guidance) for details on using Azure storage account for training in Azure Machine Learning. 

2. Execute pre-training

    The BERT pre-training job in Azure Machine Learning can be launched using either of these environments:

    * Azure Machine Learning [Compute Instance](https://docs.microsoft.com/en-us/azure/machine-learning/concept-compute-instance) to run the Jupyter notebook.
    * Azure Machine Learning [SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py)

    You will need a [GPU optimized compute target](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets#amlcompute) - _either NCv3 or NDv2 series_, to execute this pre-training job.

    Execute the steps in the Python notebook [azureml-notebooks/run-pretraining.ipynb](azureml-notebooks/run-pretraining.ipynb) within your environment. If you have a local setup to run an Azure ML notebook, you could run the steps in the notebook in that environment. Otherwise, a compute instance in Azure Machine Learning could be created and used to run the steps.

## BERT pre-training with ONNX Runtime directly on ND40rs_v2 

1. Check pre-requisites

    * CUDA 10.1
    * Docker
    * [NVIDIA docker toolkit](https://github.com/NVIDIA/nvidia-docker)

2. Build the ONNX Runtime Docker image

    Build the onnxruntime wheel from source into a Docker image.
    ```bash
    cd nvidia-bert/docker
    bash build.sh
    cd ../..
    ```    
    - Tag this image __onnxruntime-bert__`
    
    To build and install the onnxruntime wheel on the host machine, follow steps [here](https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#Training)

3. Set correct paths to training data for docker image.

   Edit `nvidia-bert/docker/launch.sh`.

   ```bash
   ...
   -v <replace-with-path-to-phase1-hdf5-training-data>:/data/128
   -v <replace-with-path-to-phase2-hdf5-training-data>:/data/512
   ...
   ```

   The two directories must contain the hdf5 training files.

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

    train_batch_size=${1:-16320}
    learning_rate=${2:-"6e-3"}
    warmup_proportion=${5:-"0.2843"}
    train_steps=${6:-7038}
    accumulate_gradients=${10:-"true"}
    gradient_accumulation_steps=${11:-340}

    train_batch_size_phase2=${17:-8160}
    learning_rate_phase2=${18:-"4e-3"}
    warmup_proportion_phase2=${19:-"0.128"}
    train_steps_phase2=${20:-1563}
    gradient_accumulation_steps_phase2=${11:-1020}
    ```
    The above defaults are tuned for an Azure NC24rs_v3.

    The training batch size refers to the number of samples a single GPU sees before weights are updated. The training is performed over _local_ and _global_ steps. A local step refers to a single backpropagation execution on the model to calculate its gradient. These gradients are accumulated every local step until weights are updated in a global step. The _microbatch_ size is samples a single GPU sees in a single backpropagation execution step. The microbatch size will be the training batch size divided by gradient accumulation steps.
    
    Note: The effective batch size will be (number of GPUs) x train_batch_size (per GPU). In general we recommend setting the effective batch size to ~64,000 for phase 1 and ~32,000 for phase 2. The number of gradient accumulation steps should be minimized without overflowing the GPU memory (i.e. maximizes microbatch size).

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
