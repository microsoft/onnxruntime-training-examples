FROM ptebic.azurecr.io/internal/azureml/aifx/nightly-ubuntu2004-cu117-py38-torch210dev:latest

RUN pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu118
RUN pip install accelerate datasets transformers
# RUN pip install git+https://github.com/huggingface/diffusers
RUN git clone https://github.com/prathikr/diffusers.git && \
        cd diffusers && \
        git checkout prathikrao/ortmodule-stablediffusionpipeline && \
        pip install .
RUN pip install azureml-core

RUN pip list