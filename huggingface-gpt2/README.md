# Accelerate GPT2 fine-tuning with ONNX Runtime Training 

This example uses ONNX Runtime Training to fine-tune the GPT2 PyTorch model maintained at https://github.com/huggingface/transformers.

You can run the training in Azure Machine Learning or in other environments.

## Setup

1. Clone this repo

    ```bash
    git clone https://github.com/microsoft/onnxruntime-training-examples.git
    cd onnxruntime-training-examples/huggingface-gpt2
    ```

2. Clone download code and model from the [HuggingFace](https://github.com/huggingface/transformers) repo

    ```bash
    git clone https://github.com/huggingface/transformers.git
    cd transformers/
    git checkout 9a0a8c1c6f4f2f0c80ff07d36713a3ada785eec5
    ```

3. Update with ORT changes

    ```bash
    git apply ../ort_addon/src_changes.patch
    cp -r ../ort_addon/ort_supplement/* ./
    cd ..
    ```

4. Build the Docker image
    
    Install the dependencies of the transformer examples and modified transformers into the base ORT Docker image.

    ```bash
    docker build --network=host -f docker/Dockerfile . --rm --pull -t onnxruntime-gpt
    ```

## Download and prepare data

The following are a minimal set of instructions to download one of the datasets used for GPT2 finetuning for the language modeling task.

Download the word-level dataset [WikiText-103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) for this sample.
Refer to the readme at [transformers](https://github.com/huggingface/transformers/tree/master/examples/language-modeling#language-model-training) for additional details.

Download the data and export path as $DATA_DIR: 
```bash
    export DATA_DIR=/path/to/downloaded/data/
```

* TRAIN_FILE: `$DATA_DIR/wiki.train.tokens`
* TEST_FILE: `$DATA_DIR/wiki.test.tokens`

## GPT2 Language Modeling fine-tuning with ONNX Runtime Training in Azure Machine Learning

1. Data Transfer

    * Transfer training data to Azure blob storage

    To transfer the data to an Azure blob storage using [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest), use command:
    ```bash
    az storage blob upload-batch --account-name <storage-name> -d <container-name> -s $DATA_DIR
    ```

    You can also use [azcopy](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10) or [Azure Storage Explorer](https://azure.microsoft.com/en-us/features/storage-explorer/) to copy data.
    We recommend that you download the data in the training environment itself or in an environment from where data transfer to training environment will be fast and efficient.

    * Register the blob container as a data store
    * Mount the data store in the compute targets used for training

    Please refer to the [storage guidance](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-access-data#storage-guidance) for details on using Azure storage account for training in Azure Machine Learning. 

2. Prepare the docker image for AML

    Follow the instructions in [setup](#Setup) to build a docker image with the required dependencies installed.

    - Push the image to a container registry. You can find additional [details](https://docs.microsoft.com/en-us/azure/container-registry/container-registry-get-started-docker-cli) about tagging the image and pushing to an [Azure Container Registry](https://docs.microsoft.com/en-us/azure/container-registry/).
    
3. Execute finetuning

    The GPT2 finetuning job in Azure Machine Learning can be launched using either of these environments:

    * Azure Machine Learning [Compute Instance](https://docs.microsoft.com/en-us/azure/machine-learning/concept-compute-instance) to run the Jupyter notebook.
    * Azure Machine Learning [SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py)

    You will need a [GPU optimized compute target](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets#amlcompute) - _either NCv3 or NDv2 series_, to execute this pre-training job.

    Execute the steps in the Python notebook [azureml-notebooks/run-finetuning.ipynb](azureml-notebooks/run-finetuning.ipynb) within your environment. If you have a local setup to run an Azure ML notebook, you could run the steps in the notebook in that environment. Otherwise, a compute instance in Azure Machine Learning could be created and used to run the steps.

## GPT2 Language Modeling fine-tuning with ONNX Runtime Training in other environments

We recommend running this sample on a system with at least one NVIDIA GPU.

1. Check pre-requisites

    * CUDA 10.1
    * Docker

2. Build the docker image

    Follow the instructions in [setup](#Setup) to build a docker image with the required dependencies installed.

    The base Docker image used is `mcr.microsoft.com/azureml/onnxruntime-training:0.1-rc1-openmpi4.0-cuda10.1-cudnn7.6-nccl2.4.8`. The Docker image is tested in AzureML environment. For running the examples in other environments, building a new base Docker image may be necessary by following the directions in the [nvidia-bert sample](../nvidia-bert/README.md).

    To build and install the onnxruntime wheel on the host machine, follow steps [here](https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#Training)

3. Set correct paths to training data for docker image.

   Edit `docker/launch.sh`.

   ```bash
   ...
   DATA_DIR=<replace-with-path-to-training-data>
   ...
   ```

   The directory must contain the training and validation files.

4. Set the number of GPUs.

    Edit `transformers/scripts/run_lm_gpt2.sh`.

    ```bash
    num_gpus=4
    ```

5. Modify other training parameters as needed.

    Edit `transformers/scripts/run_lm_gpt2.sh`.

    ```bash
        --model_type=gpt2 
        --model_name_or_path=gpt2 
        --tokenizer_name=gpt2  
        --config_name=gpt2  
        --per_gpu_train_batch_size=1  
        --per_gpu_eval_batch_size=4  
        --gradient_accumulation_steps=16 
        --block_size=1024  
        --weight_decay=0.01
        --logging_steps=100 
        --num_train_epochs=5 
    ```

    Consult the huggingface transformers [training_args](https://github.com/huggingface/transformers/blob/master/src/transformers/training_args.py) for additional details.

6. Launch interactive container.

    ```bash
    bash docker/launch.sh
    ```

7. Launch the fine-tuning run

    ```bash
    bash /workspace/transformers/scripts/run_lm_gpt2.sh
    ```

    If you get memory errors, try reducing the batch size. You can find the recommended batch sizes for ORT [here](azureml-notebooks/run-finetuning.ipynb###Creat-Estimator).
    If the flags enabling evaluation and the evaluation data file are passed, the training is followed by evaluation and the perplexity is printed.