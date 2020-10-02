FROM mcr.microsoft.com/azureml/onnxruntime-training:0.1-rc3-openmpi4.0-cuda10.2-cudnn8.0-nccl2.7
WORKDIR /workspace
RUN pip install azureml azureml-core
COPY transformers/ transformers/
RUN apt-get update && \
    apt-get install -y build-essential
RUN cd transformers/ && \
    python3 -m pip install --no-cache-dir . && \
    python3 -m pip install --no-cache-dir -r examples/requirements.txt
RUN rm -rf transformers/
CMD ["/bin/bash"]
