FROM ptebic.azurecr.io/internal/azureml/aifx/nightly-ubuntu2004-cu117-py38-torch210dev:latest

RUN pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu118

# text-to-image dependencies taken from: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/requirements.txt
RUN pip install accelerate>=0.16.0 transformers>=4.25.1 datasets ftfy tensorboard Jinja2
RUN pip install git+https://github.com/huggingface/diffusers

RUN pip install azureml-core

RUN pip list