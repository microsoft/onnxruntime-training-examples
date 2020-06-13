# Accelerate GPT2 fine-tuning with ONNX Runtime

This example uses ONNX Runtime to fine-tune the GPT2 PyTorch model maintained at https://github.com/huggingface/transformers.

You can run the training in Azure Machine Learning or locally.

## Setup

1. Clone this repo

    ```bash
    git clone https://github.com/microsoft/onnxruntime-training-examples.git
    cd onnxruntime-training-examples/huggingface-gpt2
    ```

2. Clone download code and model

    ```bash
    git clone https://github.com/huggingface/transformers.git
    cd transformers/
    git checkout 9a0a8c1c6f4f2f0c80ff07d36713a3ada785eec5
    ```

3. Update with ORT changes

    ```bash
    git apply ../ort_addon/src_changes.patch
    cp -r ../ort_addon/ort_supplement/* ./
    ```

## Download and prepare data

The following are a minimal set of instructions to download one of the datasets used for GPT2 finetuning for the language modeling task.

Refer to the readme at [transformers](https://github.com/huggingface/transformers/tree/master/examples/language-modeling#language-model-training) for additional details.
Download the word-level dataset [WikiText-103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) for this sample.

Download the data and export path as $DATA_DIR: 
```bash
    export DATA_DIR=/path/to/downloaded/data/
```

* TRAIN_FILE: `$DATA_DIR/wiki.train.tokens`
* TEST_FILE: `$DATA_DIR/wiki.test.tokens`

Below instructions refer to these hdf5 data files as the data to make accessible to training process.

## GPT2 Language Modeling fine-tuning with ONNX Runtime in Azure Machine Learning

1. Setup environment

    * Transfer training data to Azure blob storage

    To transfer the data to an Azure blob storage using [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest), use command:
    ```bash
    az storage blob upload-batch --account-name <storage-name> -d <container-name> -s $DATA_DIR
    ```

    * Register the blob container as a data store
    * Mount the data store in the compute targets used for training

    Please refer to the [storage guidance](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-access-data#storage-guidance) for details on using Azure storage account for training in Azure Machine Learning. 

2. Build the docker image

    Install the dependencies of the transformer examples and modified transformers into the base ORT Docker image.
    ```bash
    cd transformers
    docker build --network=host -f docker/transformers-ort-gpu/Dockerfile . --rm --pull -t onnxruntime-gpt
    cd ../..
    ```    
    - Push the image to a container registry. You can find additional [details](https://docs.microsoft.com/en-us/azure/container-registry/container-registry-get-started-docker-cli) about tagging the image and pushing to an [Azure Container Registry](https://docs.microsoft.com/en-us/azure/container-registry/).
    
2. Execute pre-training

    The GPT2 finetunin job in Azure Machine Learning can be launched using either of these environments:

    * Azure Machine Learning [Compute Instance](https://docs.microsoft.com/en-us/azure/machine-learning/concept-compute-instance) to run the Jupyter notebook.
    * Azure Machine Learning [SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py)

    You will need a [GPU optimized compute target](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets#amlcompute) - _either NCv3 or NDv2 series_, to execute this pre-training job.

    Execute the steps in the Python notebook [azureml-notebooks/run-finetuning.ipynb](azureml-notebooks/run-finetuning.ipynb) within your environment. If you have a local setup to run an Azure ML notebook, you could run the steps in the notebook in that environment. Otherwise, a compute instance in Azure Machine Learning could be created and used to run the steps.

## GPT2 Language Modeling fine-tuning with ONNX Runtime locally

1. Check pre-requisites

    * CUDA 10.1
    * Docker

2. Build the ONNX Runtime Docker image

    Install the dependencies of the transformer examples and modified transformers into the base ORT Docker image.
    ```bash
    cd transformers
    docker build --network=host -f docker/transformers-ort-gpu/Dockerfile . --rm --pull -t onnxruntime-gpt
    cd ../..
    ```    

    To build and install the onnxruntime wheel on the host machine, follow steps [here](https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#Training)

3. Set correct paths to training data for docker image.

   Edit `huggingface-gpt2/docker/launch.sh`.

   ```bash
   ...
   DATA_DIR=<replace-with-path-to-training-data>
   ...
   ```

   The directory must contain the training and validation files.

4. Launch interactive container.

    ```bash
    cd workspace/BERT
    bash ../../nvidia-bert/docker/launch.sh
    ```

5. Set the number of GPUs and switch for ORT or PyTorch run.

    Edit `workspace/transformers/scripts/run_lm_gpt2.sh`.

    ```bash
    num_gpus=4
    use_ort=true
    ```

5. Modify other training parameters as needed.

    Edit `workspace/transformers/scripts/run_lm_gpt2.sh`.

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

5. Launch the fine-tuning run

    ```bash
    bash /workspace/transformers/scripts/run_lm_gpt2.sh
    ```

    If you get memory errors, try reducing the batch size. You can find the recommended batch sizes for ORT and PyTorch [here](azureml-notebooks/run-finetuning.ipynb###Creat-Estimator).









## Fine-tuning

For fine-tuning tasks, follow [model_evaluation.md](model_evaluation.md)
