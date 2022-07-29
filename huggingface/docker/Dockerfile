FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu115-py38-torch1110
ENV DISABLE_MLFLOW_INTEGRATION TRUE

RUN pip3 install accelerate
RUN pip3 install datasets>=1.8.0
RUN pip3 install sentencepiece>0.1.92
RUN pip3 install protobuf
RUN pip3 install sacrebleu>=1.4.12
RUN pip3 install py7zr
RUN pip3 install git+https://github.com/huggingface/optimum.git
RUN pip3 install git+https://github.com/huggingface/transformers
# set up and configure ORT==1.12.0 so that RoBERTa can be run
RUN pip3 uninstall onnxruntime-training
RUN python -m pip install onnxruntime-training==1.12.0 -f https://download.onnxruntime.ai/onnxruntime_stable_cu115.html
RUN python -m torch_ort.configure
