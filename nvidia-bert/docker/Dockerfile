# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

FROM nvcr.io/nvidia/pytorch:20.03-py3

# Environment variables
ENV STAGE_DIR=/root/gpu/install \
    CUDNN_DIR=/usr/local/cudnn \
    CUDA_DIR=/usr/local/cuda-10.1 \
    CUDA_HOME=/usr/local/cuda \
    CUDNN_HOME=/usr/lib/x86_64-linux-gnu \
    CUDACXX=/usr/local/cuda/bin/nvcc

RUN mkdir -p $STAGE_DIR

RUN apt-get -y update && \
    apt-get -y install \
      build-essential \
      autotools-dev \
      rsync \
      curl \
      wget \
      jq \
      openssh-server \
      openssh-client \
    # No longer in 'minimal set of packages'
      sudo \
    # Needed by OpenMPI
      cmake \
      g++ \
      gcc \
    # ifconfig
      net-tools \
      binutils \
      libjpeg-turbo8 \
	  libsqlite3-0 \
	  vim \
	  git \
	  libgd-tools \
	  libgd3 \
	  python2.7 \
	  python3 \
	  apt \
	  ca-certificates \
      libssl1.1 && \
    apt-get autoremove


# Labels for the docker
LABEL description="This docker sets up the environment to run Nvidia bert with ORT Training"

# Everything above this line is required for the docker.
# Add your personalized features below here.

##
# Infiniband setup
ENV MOFED_VERSION=4.7-3.2.9.0
ENV MOFED_OS=ubuntu18.04
ENV MOFED_FILENAME=MLNX_OFED_LINUX-${MOFED_VERSION}-${MOFED_OS}-x86_64
RUN curl -fSsL https://www.mellanox.com/downloads/ofed/MLNX_OFED-${MOFED_VERSION}/${MOFED_FILENAME}.tgz |    tar -zxpf -
RUN cd MLNX_OFED_LINUX-${MOFED_VERSION}-${MOFED_OS}-x86_64 && \
    ./mlnxofedinstall --force --user-space-only --without-fw-update && \
    cd .. && \
    rm -r MLNX_OFED_LINUX-${MOFED_VERSION}-${MOFED_OS}-x86_64

# ORT requirement

WORKDIR $STAGE_DIR
ENV OPENMPI_BASEVERSION=4.0
ENV OPENMPI_VERSION=${OPENMPI_BASEVERSION}.2
RUN cd ${STAGE_DIR} && \
    wget -q -O - https://download.open-mpi.org/release/open-mpi/v${OPENMPI_BASEVERSION}/openmpi-${OPENMPI_VERSION}.tar.gz | tar xzf - && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --enable-mca-no-build=btl-uct --prefix=/usr/local/openmpi-${OPENMPI_VERSION} && \
    make -j"$(nproc)" install && \
    ln -s /usr/local/openmpi-${OPENMPI_VERSION} /usr/local/mpi4 && \
    # Sanity check:
    test -f /usr/local/mpi4/bin/mpic++ && \
    cd ${STAGE_DIR} && \
    rm -r ${STAGE_DIR}/openmpi-${OPENMPI_VERSION}
ENV PATH=/usr/local/mpi4/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi4/lib:/usr/local/mpi4/lib64:${LD_LIBRARY_PATH}
# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/mpi4/bin/mpirun /usr/local/mpi4/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/mpi4/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root --prefix /usr/local/mpi4 "$@"' >> /usr/local/mpi4/bin/mpirun && \
    chmod a+x /usr/local/mpi4/bin/mpirun

ENV CMAKE_VERSION=3.16.4
RUN cd /usr/local && \
    wget -q -O - https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz | tar zxf -
ENV PATH=/usr/local/cmake-${CMAKE_VERSION}-Linux-x86_64/bin:${PATH}

RUN apt-get -y update && \
    apt-get install -y bc tar sudo git bash

RUN apt-get update && apt-get install -y pbzip2 pv bzip2 cabextract

WORKDIR /workspace

# additional dependencies for scripts
RUN pip install --no-cache-dir \
    tqdm boto3 requests six ipdb \
    h5py html2text nltk progressbar \
    mpi4py git+https://github.com/NVIDIA/dllogger

# add pytorch patch for opset 10
COPY pyt_patch /tmp/pyt_patch
RUN cp /tmp/pyt_patch/symbolic_opset10.py /opt/conda/lib/python3.6/site-packages/torch/onnx/

# Build onnxruntime training wheel from orttraining_rc1 branch
RUN git clone https://github.com/microsoft/onnxruntime.git &&\
    cp onnxruntime/ThirdPartyNotices.txt ThirdPartyNotices.txt &&\
    cp onnxruntime/dockerfiles/LICENSE-IMAGE.txt LICENSE-IMAGE.txt &&\
    cd onnxruntime &&\
    git checkout orttraining_rc1_perf &&\
    python tools/ci_build/build.py \
        --config RelWithDebInfo \
        --use_cuda \
        --cuda_home /usr/local/cuda \
        --cudnn_home /usr/lib/x86_64-linux-gnu/ \
        --update \
        --build \
        --build_wheel \
        --enable_training \
        --parallel \
        --build_dir build \
        --skip_tests \
        --cmake_extra_defines ONNXRUNTIME_VERSION=`cat ./VERSION_NUMBER` &&\
    pip install \
          build/RelWithDebInfo/dist/*.whl &&\
    cd .. &&\
    rm -rf onnxruntime /opt/cmake

RUN apt-get install -y iputils-ping
RUN ldconfig