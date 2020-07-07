# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
FROM onnxruntime-pytorch-for-examples

# install nvidia run script dependencies
RUN apt-get install -y git &&\
    pip install --no-cache-dir \
    tqdm boto3 requests six ipdb \
    h5py html2text nltk progressbar \
    git+https://github.com/NVIDIA/dllogger
