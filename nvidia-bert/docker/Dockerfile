# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
FROM mcr.microsoft.com/azureml/onnxruntime-training:0.1-rc3.1-openmpi4.0-cuda10.2-cudnn8.0-nccl2.7

# install nvidia run script dependencies
RUN apt-get -y update &&\
    apt-get install -y git &&\
    pip install --no-cache-dir \
    tqdm boto3 requests six ipdb \
    h5py html2text nltk progressbar \
    git+https://github.com/NVIDIA/dllogger
