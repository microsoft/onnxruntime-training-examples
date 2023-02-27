FROM mcr.microsoft.com/devcontainers/miniconda:0-3

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

# [Optional] Uncomment to install a different version of Python than the default
RUN conda install -y python=3.9

RUN conda install -n base ipykernel --update-deps --force-reinstall

RUN pip install numpy onnx pillow
RUN pip install torch==1.13.1+cpu torchvision -f https://download.pytorch.org/whl/torch/
RUN pip install onnxruntime-training --pre -f https://download.onnxruntime.ai/onnxruntime_nightly_cpu.html

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
