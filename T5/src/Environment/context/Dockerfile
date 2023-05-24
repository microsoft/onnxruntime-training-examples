FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.0-cuda11.7

RUN pip install azureml-evaluate-mlflow
RUN pip install git+https://github.com/huggingface/accelerate
RUN pip install evaluate datasets rouge-score nltk py7zr optimum
RUN pip install git+https://github.com/huggingface/transformers
RUN pip install  git+https://github.com/huggingface/peft.git
RUN pip install diffusers==0.16.1