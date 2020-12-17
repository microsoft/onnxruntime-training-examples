FROM mcr.microsoft.com/azureml/onnxruntime-training:0.1-rc3.1-openmpi4.0-cuda10.2-cudnn8.0-nccl2.7
WORKDIR .
RUN python3 -m pip install --no-cache-dir torch torchtext sympy
CMD ["/bin/bash"]
