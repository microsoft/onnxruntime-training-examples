#!/bin/bash
set -eo pipefail

# clone nvidia examples repository and checkout specific commit
git clone git@github.com:NVIDIA/DeepLearningExamples.git nvidia-examples &&
    cd nvidia-examples && 
    git checkout 4733603577080dbd1bdcd51864f31e45d5196704 && 
    cd ..

# keep only bert using pytorch example directory
mkdir -p workspace && 
    cp -r nvidia-examples/PyTorch/LanguageModeling/BERT/* workspace/ &&
    rm -fr nvidia-examples

# add onnxruntime and azureml files into nvidia example
cp -r ort_patch/* workspace/
