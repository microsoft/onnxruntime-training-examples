# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Todo: Fill with official rc3 release image
FROM ortrelease.azurecr.io/onnxruntime-training:orttraining_rc3_latest

# install nvidia run script dependencies
RUN apt-get -y update &&\
    apt-get install -y git &&\
    pip install --no-cache-dir \
    tqdm boto3 requests six ipdb \
    h5py html2text nltk progressbar \
    git+https://github.com/NVIDIA/dllogger
