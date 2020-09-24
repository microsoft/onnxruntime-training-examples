FROM ortrelease.azurecr.io/onnxruntime-training:orttraining_rc3_latest
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
