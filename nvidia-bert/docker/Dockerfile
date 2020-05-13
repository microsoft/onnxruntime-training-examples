# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

FROM mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04
RUN apt update -y && pip install --upgrade pip

# set paths for cmake, miniconda, and cuda
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/cmake/bin:/opt/miniconda/bin:${PATH}
ENV LD_LIBRARY_PATH /opt/miniconda/lib:$LD_LIBRARY_PATH

# install pytorch stable using CUDA 10.1
RUN pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# add pytorch patch for opset 10
COPY pyt_patch /tmp/pyt_patch
RUN cp /tmp/pyt_patch/symbolic_opset10.py /opt/miniconda/lib/python3.7/site-packages/torch/onnx/

# build and install onnxruntime
WORKDIR /src
RUN wget --quiet https://github.com/Kitware/CMake/releases/download/v3.14.3/cmake-3.14.3-Linux-x86_64.tar.gz &&\
    tar zxf cmake-3.14.3-Linux-x86_64.tar.gz &&\
    mv cmake-3.14.3-Linux-x86_64 /opt/cmake &&\
    rm -rf cmake-3.14.3-Linux-x86_64.tar.gz

RUN git clone https://github.com/microsoft/onnxruntime.git &&\
    cd /src/onnxruntime &&\
    git checkout orttraining_rc1 &&\
    /bin/sh ./build.sh \
        --config RelWithDebInfo \
        --use_cuda \
        --cuda_home /usr/local/cuda \
        --cudnn_home /usr/lib/x86_64-linux-gnu/ \
        --update \
        --build \
        --build_wheel \
        --enable_training \
        --parallel \
        --cmake_extra_defines ONNXRUNTIME_VERSION=`cat ./VERSION_NUMBER`
RUN pip install \
          /src/onnxruntime/build/Linux/RelWithDebInfo/dist/*.whl

# build and install apex
RUN git clone https://github.com/NVIDIA/apex && cd apex &&\
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# install additional dependencies for scripts
RUN apt-get update && apt-get install -y pbzip2 pv bzip2 cabextract
RUN pip install --no-cache-dir \
 tqdm boto3 requests six ipdb h5py html2text nltk progressbar mpi4py \
 git+https://github.com/NVIDIA/dllogger

RUN apt-get install -y iputils-ping
RUN ldconfig

