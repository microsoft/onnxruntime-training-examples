Instructions:

1. Run "./setup_workspace.sh" to create workspace/
This downloads the NVIDIA PyTorch BERT example and adds in files to use onnxruntime as backend.

2. cd workspace and run "bash scripts/docker/build_ort.sh"
This builds onnxruntime from source and contains CUDA 10.1, MPI, and PyTorch 1.5.

3. See "Getting the data" section of workspace/README.md by NVIDIA to download training data.
This will require running workspace/data/create_datasets_from_start.sh within their container.

For local run,

    4. Update workspace/scripts/docker/launch_ort.sh to mount training data:
    "-v <replace-with-path-to-phase1-hdf5-training-data>:/data/128 
     -v <replace-with-path-to-phase2-hdf5-training-data>:/data/512"
    From workspace root launch using "bash scripts/docker/launch_ort.sh"

    5. Set 'num_gpus' at top of scripts/run_pretraining_ort.sh
    Withing container run "bash scripts/run_pretraining_ort.sh".

    Feel free to adjust other parameters at top of scripts/run_pretraining_ort.sh
    For more details, consult workspace/README.md by NVIDIA in reference to scripts/run_pretraining.sh.

For Azure run,

    4. Install Azure Cli and Azure ML CLI and SDK
    Consult https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest

    curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
    az login
    az extension add -n azure-cli-ml
    pip install --upgrade azureml-sdk

    4. Create Azure machine learning workspace.
    Consult https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py
            https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace-cli

    az group create --name <resource-group-name> --location <location>
    az ml workspace create -w <workspace-name> -g <resource-group-name>

    5. Create Azure container registry and upload Docker image.
    Consult https://docs.microsoft.com/en-us/azure/container-registry/container-registry-get-started-docker-cli

    az acr create --name <acr-name> --resource-group <resource-group-name> --sku <sku-type>
    az acr login --name <acr-name>
    docker tag bert-onnxruntime <acr-name>.azurecr.io/bert-onnxruntime
    docker push <acr-name>.azurecr.io/bert-onnxruntime

    6. Create storage and upload training data.
    Consult https://docs.microsoft.com/en-us/azure/storage/common/storage-account-create?tabs=azure-portal
            https://docs.microsoft.com/en-us/cli/azure/storage/blob?view=azure-cli-latest#az-storage-blob-upload-batch

    az storage account create --resource-group <my-resource-group> --name <storage-name>
    az storage container create --account-name <storage-name> --name <container-name>
    az storage blob upload-batch --account-name <storage-name>  -d <container-name>  -s <path-to-training-data>

    7. Follow further instructions in azureml-notebooks/azure-pretraining.ipynb.