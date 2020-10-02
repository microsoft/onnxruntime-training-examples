# Accelerate BERT pre-training with ONNX Runtime

This example uses ONNX Runtime to pre-train the BERT PyTorch model maintained at https://github.com/NVIDIA/DeepLearningExamples.

You can run the training in Azure Machine Learning or on an Azure VM with NVIDIA GPU.

## Download and prepare data

The following are a minimal set of instructions to download and process the Wiki dataset used for BERT pre-training.

To include additional datasets, and for more details, refer to [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#getting-the-data).

Note that the datasets used for BERT pre-training need a large amount of disk space. After processing, the data should be made available for training. Due to the large size of the data copy, we recommend that you execute the steps below in the training environment itself or in an environment from where data transfer to training environment will be fast and efficient. Be advised the training data preparation can take approximately 40 hours.

1. Clone this repo

    ```bash
    git clone https://github.com/microsoft/onnxruntime-training-examples.git
    cd onnxruntime-training-examples
    ```

2. Check pre-requisites

    * Python 3.6 invoked as `python`
    * Natural Language Toolkit (NLTK) `pip install nltk`
    * HDF5 library `pip install h5py`
    * Amazon Web Services SDK `pip install boto3`
    * HTTP Requests `pip install requests`

3. Clone download code

    ```bash
    git clone --no-checkout https://github.com/NVIDIA/DeepLearningExamples.git
    cd DeepLearningExamples/
    git checkout 4733603577080dbd1bdcd51864f31e45d5196704
    cd ..
    ```

4. Create working directory for data download and formatting operations

    ```bash
    mkdir -p workspace
    mv DeepLearningExamples/PyTorch/LanguageModeling/BERT/ workspace
    rm -rf DeepLearningExamples
    cd workspace
    git clone https://github.com/attardi/wikiextractor.git
    cd wikiextractor/
    git checkout e4abb4cbd019b0257824ee47c23dd163919b731b
    cd ../../ 
    ```

5. Download and prepare Wikicorpus training data in HDF5 format.

    ```bash
    # Run all the below commands step-by-step from the root of the repository
    # The expected times reflect West US Azure VM with Xeon(R) CPU E5-2673 v4 @ 2.30GHz

    export BERT_PREP_WORKING_DIR=./workspace/BERT/data/

    # Download google_pretrained_weights
    python ./workspace/BERT/data/bertPrep.py --action download --dataset google_pretrained_weights

    # Download wikicorpus_en via wget (approx 2 hours)
    mkdir -p ./workspace/BERT/data/download/wikicorpus_en
    cd ./workspace/BERT/data/download/wikicorpus_en
    wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
    bzip2 -dv enwiki-latest-pages-articles.xml.bz2
    mv enwiki-latest-pages-articles.xml wikicorpus_en.xml
    cd ../../../../..

    # Fix path issue to use BERT_PREP_WORKING_DIR as prefix for path instead of hard-coded prefix
    sed -i "s/path_to_wikiextractor_in_container = '/path_to_wikiextractor_in_container = './g" ./workspace/BERT/data/bertPrep.py

    # Format text files (approx 90 min using four processes)
    python ./workspace/BERT/data/bertPrep.py --action text_formatting --dataset wikicorpus_en

    # Shard text files (approx 20 hours)
    python ./workspace/BERT/data/bertPrep.py --action sharding --dataset wikicorpus_en

    # Fix path to workspace to allow running outside of the docker container
    sed -i "s/python \/workspace\/bert/python .\/workspace\/BERT/g" ./workspace/BERT/data/bertPrep.py

    # Create HDF5 files Phase 1 (approx 450 min using four processes)
    python ./workspace/BERT/data/bertPrep.py --action create_hdf5_files --dataset wikicorpus_en --max_seq_length 128 \
      --max_predictions_per_seq 20 --vocab_file ./workspace/BERT/data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt --do_lower_case 1

    # Create HDF5 files Phase 2 (approx 450 min using four processes)
    python ./workspace/BERT/data/bertPrep.py --action create_hdf5_files --dataset wikicorpus_en --max_seq_length 512 \
    --max_predictions_per_seq 80 --vocab_file ./workspace/BERT/data/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt --do_lower_case 1
    ```

6. Make data accessible for training

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

## BERT pre-training with ONNX Runtime directly on NC24rs_v3 (or similar NVIDIA capable Azure VM) 

1. Check pre-requisites

    * CUDA 10.2
    * Docker
    * [NVIDIA docker toolkit](https://github.com/NVIDIA/nvidia-docker)

2. Build the ONNX Runtime Docker image

    Build the onnxruntime wheel from source into a Docker image.
    ```bash
    cd nvidia-bert/docker
    bash build.sh
    cd ../..
    ```    
    - Tag this image __onnxruntime-pytorch-for-bert__`
    
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

4. Set the number of GPUs and other training parameters.

    Edit `nvidia-bert/run_pretraining.sh`.

    ```bash
    # computation
    num_gpus=${1:-4}
    gpu_feed_batch_size=${2:-48}
    gradient_accumulation_passes=${3:-16}
    precision=${4:-"fp16"}
    allreduce_post_accumulation="true"
    deepspeed_zero_stage="false"
    learning_rate="6e-3"
    warmup_proportion="0.2843"

    # administrative
    path_to_phase1_training_data=/data/128
    path_to_phase2_training_data=/data/512
    phase="phase1"
    training_steps=${5:-400}
    seed=${7:-$RANDOM}
    results_dir=./results
    create_logfile="true"
    debug_output="false"
    init_checkpoint="None"
    skip_checkpointing="false"
    save_checkpoint_interval=${6:-200}
    resume_from_step=0
    smooth_output_passes=32
    bert_config=bert_config.json
    ```
    The above values are for an example run on an Azure NC24rs_v3.

    The training is performed over _local_ passes and _global_ steps. A local pass refers to a single backpropagation execution on the model to calculate its gradient. The GPU feed batch size refers to the number of samples fed in one local pass. The gradients are accumulated each local pass until weights are updated in a global step. 

    Note: The effective global batch size will be (number nodes) x (number GPUs per node) x (gradient accumulation passes). In general it is recommend setting the global batch size to ~64,000 for phase 1 and ~32,000 for phase 2. The number of gradient accumulation steps should be minimized without overflowing the GPU memory (i.e. maximizes GPU feed batch size).

5. Launch interactive container.

    ```bash
    cd nvidia-bert
    bash ./docker/launch.sh
    ```

7. Launch pre-training run

    ```bash
    bash run_pretraining.sh
    ```

    If you get memory errors, try reducing the batch size.

## Fine-tuning

For fine-tuning tasks, follow [model_evaluation.md](model_evaluation.md)
