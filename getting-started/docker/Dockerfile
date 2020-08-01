FROM mcr.microsoft.com/azureml/onnxruntime-training:0.1-rc2-openmpi4.0-cuda10.2-cudnn7.6-nccl2.7.6
WORKDIR .
RUN python3 -m pip install --no-cache-dir torch torchtext
CMD ["/bin/bash"]
