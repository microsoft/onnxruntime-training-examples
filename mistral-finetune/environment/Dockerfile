FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu118-py38-torch211

RUN pip uninstall onnxruntime-training -y && \
    pip install onnxruntime-training --pre -f https://download.onnxruntime.ai/onnxruntime_nightly_cu118.html && \
    TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX" python -m onnxruntime.training.ortmodule.torch_cpp_extensions.install

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN pip list
