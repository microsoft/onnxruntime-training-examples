# Get started with ONNX Runtime Training

Train a PyTorch transformer model with ONNX Runtime.

This sample trains the model defined here: https://pytorch.org/tutorials/beginner/transformer_tutorial.html. This is a transformer model composed of PyTorch transformer components such as the PositionalEncoder, TransformerEncoder and word Embedding.

The language model assigns a probability for the likelihood of a given word (or a sequence of words) to follow a sequence of words.

The Wikitext-2 dataset (downloaded using `torchtext.dataset`) is used to train the model.

You can either run this sample using the provided docker image, which contains the ONNX Runtime Training module along with all of its dependencies, or in standalone mode by cloning and building ONNX Runtime Training from source.

The purpose of this sample is to demonstrate the concepts and usage of ONNX Runtime Training, rather than to show its performance capabilities. For more complex samples that do demonstrate improved training performance at scale, please refer to the other samples in this repo:

* [NVIDIA BERT pre-training](../nvidia-bert)
* [HuggingFace GPT-2 fine-tuning](../huggingface-gpt2)

## Train with docker

## Train standalone

### Dependencies

* Python 3.7
* torch
* torchtext
* onnx
* onnxruntime-gpu built for training

### Train the model

* The model is defined in model.py
* Run `python train.py` to train with PyTorch
* Run `python train_ort.py` to train with ORT
* Run `python predict.py` to make a predicton (incomplete)
