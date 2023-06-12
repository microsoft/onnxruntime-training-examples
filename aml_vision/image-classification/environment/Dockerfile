FROM ptebic.azurecr.io/public/azureml/aifx/stable-ubuntu2004-cu117-py38-torch1131:ort1.15.0-vision-patch
RUN pip install accelerate datasets evaluate optimum transformers
RUN pip install azureml-core scikit-learn

RUN pip list
