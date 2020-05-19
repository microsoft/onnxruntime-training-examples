# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
FROM nvidia/cuda:10.1-cudnn7-devel

RUN apt-get update &&\
    apt-get install -y sudo git bash

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget zip ca-certificates build-essential curl libcurl4-openssl-dev \
    libssl-dev python3-dev python3.6 python3-pip python3-setuptools

# set paths for cmake, miniconda, and cuda
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/cmake/bin:/opt/miniconda/bin:${PATH}
ENV LD_LIBRARY_PATH /opt/miniconda/lib:$LD_LIBRARY_PATH

# install pytorch stable using CUDA 10.1
RUN pip3 install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# add pytorch patch for opset 10
COPY pyt_patch /tmp/pyt_patch
RUN cp /tmp/pyt_patch/symbolic_opset10.py /usr/local/lib/python3.6/dist-packages/torch/onnx/

# build and install onnxruntime
WORKDIR /src
RUN wget --quiet https://github.com/Kitware/CMake/releases/download/v3.14.3/cmake-3.14.3-Linux-x86_64.tar.gz &&\
    tar zxf cmake-3.14.3-Linux-x86_64.tar.gz &&\
    mv cmake-3.14.3-Linux-x86_64 /opt/cmake &&\
    rm -rf cmake-3.14.3-Linux-x86_64.tar.gz

RUN git clone https://github.com/microsoft/onnxruntime.git &&\
    cp /src/onnxruntime/ThirdPartyNotices.txt /src/ThirdPartyNotices.txt &&\
    cp /src/onnxruntime/dockerfiles/LICENSE-IMAGE.txt /src/LICENSE-IMAGE.txt &&\
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
        --cmake_extra_defines ONNXRUNTIME_VERSION=`cat ./VERSION_NUMBER` &&\
    pip3 install \
          /src/onnxruntime/build/Linux/RelWithDebInfo/dist/*.whl &&\
    cd .. &&\
    rm -rf onnxruntime /opt/cmake
          
# install additional dependencies for scripts
RUN apt-get update && apt-get install -y pbzip2 pv bzip2 cabextract
RUN pip3 install --no-cache-dir \
 tqdm boto3 requests six ipdb h5py html2text nltk progressbar mpi4py \
 git+https://github.com/NVIDIA/dllogger

RUN apt-get install -y iputils-ping
RUN ldconfig
