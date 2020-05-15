# ONNX Runtime Training Examples

This repo has examples for using [ONNX Runtime](https://github.com/microsoft/onnxruntime) for accelerating training of [Transformer](https://arxiv.org/abs/1706.03762) models. These examples focus on large scale model training and achieving the best performance in [Azure Machine Learning service](https://azure.microsoft.com/en-us/services/machine-learning/) and [NVIDIA DGX-2](https://www.nvidia.com/en-us/data-center/dgx-2). The examples demonstrate how training using ONNX Runtime integrates with training scripts implemented in PyTorch by enabling the conversion of the models defined using PyTorch's `torch.nn.module` to [`ORTTrainer`](https://github.com/microsoft/onnxruntime/blob/orttraining_rc1/orttraining/orttraining/python/ort_trainer.py#L480)* API with just a few lines of code change. 

*_A newer version of [ORTTrainer API](https://github.com/microsoft/onnxruntime/blob/orttraining_rc1/orttraining/orttraining/python/ort_trainer.py#L480) is under active development and will be released in a few months._

## Examples

Outline the examples in the repository. 

| Example       | Description                                |
|-------------------|--------------------------------------------|
| `nvidia-bert`     | Using ONNX Runtime with [BERT pretraining implementation in PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT) maintained by nvidia |

<!-- 
| `CONTRIBUTING.md` | Guidelines for contributing to the sample. |
-->
## Prerequisites

The examples in this repo depend on a Docker image (TODO: add link) that includes ONNX Runtime for training. The Docker image is tested in AzureML and DGX-2 environments. For running the examples in other environments, building a new Docker image may be necessary.

## Running the sample

Readme files for every example listed in the table above provides a step-by-step instructions to execute the example.

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
