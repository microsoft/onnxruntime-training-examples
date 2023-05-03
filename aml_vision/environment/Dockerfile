FROM ptebic.azurecr.io/test/public/azureml/aifx/stable-ubuntu2004-cu117-py38-torch1131:20230428.custom_ort_nightly
# FROM ptebic.azurecr.io/internal/azureml/aifx/nightly-ubuntu2004-cu117-py38-torch210dev:latest
RUN pip install azureml-core datasets evaluate optimum scikit-learn
RUN pip install git+https://github.com/huggingface/transformers
RUN pip install git+https://github.com/huggingface/accelerate

RUN pip list