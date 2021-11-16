

# ORTModule Examples
This example uses ORTModule to fine-tune several popular [HuggingFace](https://huggingface.co/) models.

## 1 Setup
1. Clone this repo and initialize git submodule
```bash
git clone https://github.com/microsoft/onnxruntime-training-examples.git
cd onnxruntime-training-examples
git submodule update --init --recursive
git submodule foreach git pull origin master
```
2. Make sure python 3.6+ is installed

We recommend using conda to manage python environment. If you do not have conda installed, you can follow the instruction to install conda [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). Once conda is installed, create a new python environment with 
```bash
conda create --name myenv python=3.6
```
3. Install azureml-core

Activate conda environment just created.
```bash
conda activate myenv
```
Install azureml dependency for script submission.
```bash
pip install azureml-core
```
## 2 Run on AzureML
### 2.1 Prerequisites
1. AzureML subscription is required to run this example. Either a config.json file ([How to get config.json file from Azure Portal](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment#workspace)) or subscription_id, resource_group, workspace_name information needs to be passed in through parameter.
2. The subscription should have a gpu cluster. This example was tested with GPU cluster of SKU [`Standard_ND40rs_v2`](https://docs.microsoft.com/en-us/azure/virtual-machines/ndv2-series). See this document for [creating gpu cluster](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=python).
### 2.2 Run this recipe
Download config.json file in 2.1 to `huggingface/script` directory. Or append below run script with AzureML workspace information such as `--workspace_name <your_workspace_name> --resource_group 
<resource_group> --subscription_id <your_subscription_id>`.

Here's an example to run run bert-large with ORTModule. `hf-ort.py` builds a docker image based on [dockerfile](docker/Dockerfile) and submits run script to AzureML according to model and run configuration. Default docker image uses cuda 11.1.
```bash
cd huggingface/script
python hf-ort.py --gpu_cluster_name <gpu_cluster_name> --hf_model bert-large --run_config ort
```
To run different models with different configuration, check below tables.

This table summarizes if model changes are required.
| Model                | Performance Compariso           | Model Change                                |
|----------------------|---------------------------------|---------------------------------------------|
| bart-large           | See [BART](BART.md)             | No model change required |
| bert-large           | See [BERT](BERT.md)             | No model change required |
| deberta-v2-xxlarge   | See [DeBERTa](DeBERTa.md)       | See [this commit](https://github.com/microsoft/huggingface-transformers/commit/0b2532a4f1df90858472d1eb2ca3ac4eaea42af1) |
| distilbert-base      | See [DistilBERT](DistilBERT.md) | No model change required |
| gpt2                 | See [GPT2](GPT2.md)             | No model change required |
| roberta-large        | See [RoBERTa](RoBERTa.md)       | See [this commit](https://github.com/microsoft/huggingface-transformers/commit/b25c43e533c5cadbc4734cc3615563a2304c18a2)|
| t5-large             | See [T5](T5.md)                 | See [this PR](https://github.com/microsoft/huggingface-transformers/pull/4/files) |

Here're the different configs and description that the recipe script take through `--run_config` parameter.

| Config    | Description |
|-----------|-------------|
| pt-fp16   | PyTorch mixed precision | 
| ort       | ORTModule mixed precision |
| ds_s1     | PyTorch + Deepspeed stage 1 |
| ds_s1_ort | ORTModule + Deepspeed stage 1|

Other parameters. Please also see parameters [`script/hf-ort.py`](azureml/hf-ort.py#L64)

| Name                | Description |
|---------------------|-------------|
| --model_batchsize   | Model batchsize per GPU | 
| --max_steps         | Max step that a model will run |
| --process_count     | Total number of GPUs (not GPUs per node). **Adjust this if target cluster is not 8 gpus** |
| --node_count        | Node count |
| --skip_docker_build | Skip docker build (use last built docker saved in AzureML environment) |
| --use_cu102         | Use Cuda 10.2 dockerfile |
| --local_run         | Run the model locally, azureml related parameters will be ignored |
#### Notes
- **Benchmark methodology**: We report samples/sec on [`ND40rs_v2`](https://azure.microsoft.com/en-us/pricing/details/machine-learning/) VMs (V100 32G x 8), Cuda 11, with stable release [`onnxruntime_training-1.8.0%2Bcu111-cp36-cp36m-manylinux2014_x86_64.whl`](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_stable_cu111.html). Cuda 10.2 option is also available through `--use_cu102` flag. Please check dependency details in [Dockerfile](docker/Dockerfile). We look at the metrics `stable_train_samples_per_second` in the log, which discards first step that includes setup time. Also please note since ORTModule takes some time to do initial setup, smaller `--max_steps` value may lead to longer total run time for ORTModule compared to PyTorch. However, if you want to see finetuning to finish faster, adjust `--max_steps` to a smaller value. Lastly, we do not recommend running this recipe on [`NC`] series VMs which uses old architecture (K80).
- **Cost and VM availability**: The finetuning job runs for ~1hr for default 8000 steps on [`ND40rs_v2`](https://azure.microsoft.com/en-us/pricing/details/machine-learning/) VMs, which costs $22.03/hr per run. Additional costs are [Azure container registry costs](https://azure.microsoft.com/en-us/pricing/details/container-registry/) for docker image storage, as well as [Azure Storage cost](https://azure.microsoft.com/en-us/pricing/details/storage/) for run history storage. Please note, [`ND40rs_v2`](https://azure.microsoft.com/en-us/pricing/details/machine-learning/) is **not** publicly available by default. To get it, after the subscription is created, user need to create a support ticket [here](https://ms.portal.azure.com/#blade/Microsoft_Azure_Support/HelpAndSupportBlade/overview), then ND series will be available.
- On first run, this script takes ~20 mins to submit the finetuning job due to building a new docker image from Dockerfile. The step to build docker image [`hf_ort_env.register(ws).build(ws).wait_for_completion()`](script/hf-ort.py#L147) can be skipped by passing `--skip_docker_build` if not running for the first time.
## 3 Run on local
### 3.1 Prerequisites
1. A machine that you can access with GPU. This recipe was tested on 8 x 32G V100 GPUs machine.
2. Know how many GPUs are there. This needs to be passed to parameter `--process_count`
### 3.2 Run this recipe
Build docker image. `Dockerfile` is for cuda 11.1, `Dockerfile-10.2` is for cuda 10.2.
```
cd huggingface/docker
sudo docker build -t hf-recipe-local-docker -f Dockerfile .
```
Run built docker image
* Replace `<onnxruntime-training-examples_path>` to your local **full path** to `onnxruntime-training-examples`
  * Usually it's located at `~/onnxruntime-training-examples/`
* `-v /dev/shm:/dev/shm` mounts `/dev/shm` to inside docker `/dev/shm`. Similarly `-v <onnxruntime-training-examples_path>:/onnxruntime-training-examples` mounts `<onnxruntime-training-examples_path>` to inside docker `/onnxruntime-training-examples/`
```
sudo docker run -it -v /dev/shm:/dev/shm -v <onnxruntime-training-examples_path>:/onnxruntime-training-examples --gpus all hf-recipe-local-docker
```
Run `hf-ort.py` script
* Reminder to use the number of GPUs available locally to parameter `--process_count`
* Depending on the memory available to local GPU, you might need to overwrite default batch size by passing in `--model_batchsize`
* `--local_run` runs the script locally
```
cd /onnxruntime-training-examples/huggingface/script/
python hf-ort.py --hf_model {hf_model} --run_config {run_config} --process_count <process_count> --local_run
```

## FAQ
### Problem with Azure Authentication
If there's an Azure authentication issue, install Azure CLI [here](https://docs.microsoft.com/en-us/cli/azure/) and run `az login --use-device-code`

### In case of `RuntimeError: CUDA out of memory` error
The issue is most likely caused by hitting a HW limitation on the target, this can be mitigated by using the following switches

`--model_batchsize` <parameter> - Change to smaller batchsize
  
`--process_count` <parameter> - Change the number of GPUs to activate 
  
#### For example
  ```
python hf-ort.py --hf_model bart-large --run_config pt-fp16 --process_count 1 --local_run --model_batchsize 1 --max_steps 20
```
