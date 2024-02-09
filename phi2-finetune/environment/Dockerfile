FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu118-py38-torch220

RUN pip uninstall onnxruntime-training -y && \
    pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT/pypi/simple/ onnxruntime-training && \
    TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX" python -m onnxruntime.training.ortmodule.torch_cpp_extensions.install

RUN pip install -U datasets evaluate accelerate scikit-learn transformers==4.36.2
RUN pip install git+https://github.com/huggingface/optimum.git

RUN pip install einops
RUN pip install --upgrade pytest
RUN pip install peft

RUN pip list
