FROM mcr.microsoft.com/azureml/aifx/stable-ubuntu2004-cu115-py38-torch1110
ENV DISABLE_MLFLOW_INTEGRATION TRUE

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
