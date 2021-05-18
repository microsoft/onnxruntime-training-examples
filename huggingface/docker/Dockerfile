# ONNX Runtime Training Module for PyTorch
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Usage
#   Build: docker build -f ort-11.1.1-cudnn8-devel-ubuntu18.04-dockerfile -t [image-name] .
#   Run: docker run -it --gpus all --name [run-name] [image-name]:latest /bin/bash
# Example:
#   docker build -f ort-11.1.1-cudnn8-devel-ubuntu18.04-dockerfile -t ort.cuda11.1 .
#   docker run -it --gpus all --name my-experiments ort.cuda11.1:latest /bin/bash

# CUDA development image for building sources
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04 as builder

# Install and update tools to minimize security vulnerabilities
RUN apt-get update
RUN apt-get install -y software-properties-common wget apt-utils patchelf git libprotobuf-dev protobuf-compiler cmake
RUN unattended-upgrade
RUN apt-get autoremove -y

# Python and pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN apt-get install -y python3-pip
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
RUN pip install --upgrade pip

# PyTorch
RUN pip install onnx ninja
RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# ORT Module
RUN pip install --pre onnxruntime-training -f https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_nightly_cu111.html
RUN pip install torch-ort

WORKDIR /stage

#Install huggingface transformers
RUN cd /stage && git clone https://github.com/microsoft/huggingface-transformers.git &&\
    cd huggingface-transformers &&\
    git checkout raviskolli/ort_t5 &&\
    pip install -e .

# Install AzureML support and commonly used packages.
RUN pip install azureml-defaults wget fairscale
RUN pip install sacrebleu datasets deepspeed