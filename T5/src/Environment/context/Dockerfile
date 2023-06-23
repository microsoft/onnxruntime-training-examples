FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.0-cuda11.7

RUN pip install accelerate==0.19.0 \
                datasets==2.12.0 \
                evaluate==0.4.0 \
                transformers==4.29.2 \
                optimum==1.8.6 \
                rogue-score \
                nltk
# RUN pip install azureml-evaluate-mlflow