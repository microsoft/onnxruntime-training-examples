# ONNX Runtime Training Examples

This repo has examples for using [ONNX Runtime](https://github.com/microsoft/onnxruntime) (ORT) for accelerating training of [Transformer](https://arxiv.org/abs/1706.03762) models. These examples focus on large scale model training and achieving the best performance in [Azure Machine Learning service](https://azure.microsoft.com/en-us/services/machine-learning/) and [NVIDIA DGX-2](https://www.nvidia.com/en-us/data-center/dgx-2). ONNX Runtime has the capability to train existing PyTorch models (implemented using `torch.nn.Module`) through its optimized backend. The examples in this repo demonstrate how [`ORTTrainer`](https://github.com/microsoft/onnxruntime/blob/orttraining_rc1/orttraining/orttraining/python/ort_trainer.py#L480)* API can be used to switch the training backend for such models to ONNX Runtime with just a few changes to the existing training code. 

*_[ORTTrainer API](https://github.com/microsoft/onnxruntime/blob/orttraining_rc1/orttraining/orttraining/python/ort_trainer.py#L480) is experimental and expected to see significant changes in the near future. A new version of the API is under active development. The improvements to the API will provide a more seamless integration with PyTorch training that requires minimal changes in users’ training code._

## Examples

Outline the examples in the repository. 

| Example       | Description                                |
|-------------------|--------------------------------------------|
| [getting-started](getting-started)| Get started with ONNX Runtime with a simple PyTorch transformer model |
| [nvidia-bert](nvidia-bert)| Using ONNX Runtime Training with [BERT pretraining implementation in PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT) maintained by nvidia |
| [huggingface-gpt2](huggingface-gpt2)| Using ONNX Runtime Training with [GPT2 finetuning for Language Modeling in PyTorch](https://github.com/huggingface/transformers/tree/master/examples/language-modeling#language-model-training) maintained by huggingface |
| [nvidia-bioBERT](nvidia-bioBERT)| Using ONNX Runtime Training with [Bio-BERT pretraining implementation in PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT/biobert) maintained by nvidia |
<!-- 
| `CONTRIBUTING.md` | Guidelines for contributing to the sample. |
-->

## Prerequisites

The examples in this repo depend on a Docker image that includes ONNX Runtime for training. The docker image is available at `mcr.microsoft.com/azureml/onnxruntime-training`. The Docker image is tested in AzureML and DGX-2 environments. For running the examples in other environments, building a new Docker image may be necessary.

## Running the sample

Readme files for every example listed above provide a step-by-step instructions to execute the example.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
