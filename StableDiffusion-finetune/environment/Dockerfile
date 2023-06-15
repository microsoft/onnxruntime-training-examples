FROM ptebic.azurecr.io/internal/azureml/aifx/nightly-ubuntu2004-cu117-py38-torch210dev:latest

RUN pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu118
# text-to-image dependencies taken from: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/requirements.txt
RUN pip install accelerate>=0.16.0 torchvision transformers>=4.25.1 datasets ftfy tensorboard Jinja2

# temporary workaround to allow ORTModule to be passed as a parameter into StableDiffusionPipeline
# see this PR for more details: https://github.com/huggingface/diffusers/pull/3791
# RUN pip install git+https://github.com/huggingface/diffusers
RUN git clone https://github.com/prathikr/diffusers.git && \
        cd diffusers && \
        git checkout prathikrao/ortmodule-stablediffusionpipeline && \
        pip install .

RUN pip install azureml-core

RUN pip list