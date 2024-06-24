
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu118-py310-torch222

# RUN pip install -U datasets evaluate accelerate scikit-learn transformers==4.40.1
RUN pip install -U datasets evaluate accelerate scikit-learn
RUN pip install git+https://github.com/huggingface/transformers.git
RUN pip install git+https://github.com/huggingface/optimum.git


# Insall numpy 1.26.4 because numpy 2.0.0 are not supported with many libraries
RUN pip uninstall numpy -y && \
    pip install numpy==1.26.4

# Error with deepspeed=0.14.2
RUN pip uninstall deepspeed -y && \
    pip install deepspeed==0.13.1
    
RUN pip install einops
RUN pip install --upgrade pytest
RUN pip install peft


RUN pip list

