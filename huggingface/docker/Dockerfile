FROM ptebic.azurecr.io/internal/azureml/aifx/stable-ubuntu2004-cu116-py38-torch1120
ENV DISABLE_MLFLOW_INTEGRATION TRUE

RUN pip3 install accelerate
RUN pip3 install datasets>=1.8.0
RUN pip3 install sentencepiece>0.1.92
RUN pip3 install protobuf
RUN pip3 install sacrebleu>=1.4.12
RUN pip3 install py7zr
RUN pip3 install git+https://github.com/huggingface/optimum.git
RUN pip3 install git+https://github.com/huggingface/transformers
