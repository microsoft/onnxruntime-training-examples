FROM mcr.microsoft.com/azureml/onnxruntime-training:0.1-rc1-openmpi4.0-cuda10.1-cudnn7.6-nccl2.4.8
WORKDIR /workspace
RUN python3 -m pip install azureml azureml-core
COPY . transformers/
RUN cd transformers/ && \
    python3 -m pip install --no-cache-dir . && \
    python3 -m pip install --no-cache-dir -r examples/requirements.txt
RUN rm -rf transformers/
CMD ["/bin/bash"]
