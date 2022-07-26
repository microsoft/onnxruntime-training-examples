# New to Onnx
- Official ORT documentation: https://www.onnxruntime.ai/  
- Official ORT GitHub Repo: https://github.com/microsoft/onnxruntime
- Official ORT Samples Repo: https://github.com/microsoft/onnxruntime-training-examples

# What is ONNX Runtime for PyTorch

ONNX Runtime for PyTorch gives you the ability to accelerate training of large transformer PyTorch models. The training time and cost are reduced with just a one line code change.

- One line code change: ORT provides a one-line addition for existing PyTorch training scripts allowing easier experimentation and greater agility.
```python
    from torch_ort import ORTModule
    model = ORTModule(model)
```

- Flexible and extensible hardware support: The same model and API works with NVIDIA and AMD GPUs; the extensible "execution provider" architecture allow you to plug-in custom operators, optimizer and hardware accelerators.

- Faster Training: Optimized kernels provide up to 1.4X speed up in training time.

- Larger Models: Memory optimizations allow fitting a larger model such as GPT-2 on 16GB GPU, which runs out of memory with stock PyTorch.

- Composable with other acceleration libraries such as Deepspeed, Fairscale, Megatron for even faster and more efficient training

- Part of the PyTorch Ecosystem. It is available via the torch-ort python package.
 
- Built on top of highly successful and proven technologies of ONNX Runtime and ONNX format.

# ONNX Runtime Training Examples

This repo has examples for using [ONNX Runtime](https://github.com/microsoft/onnxruntime) (ORT) for accelerating training of [Transformer](https://arxiv.org/abs/1706.03762) models. These examples focus on large scale model training and achieving the best performance in [Azure Machine Learning service](https://azure.microsoft.com/en-us/services/machine-learning/). ONNX Runtime has the capability to train existing PyTorch models (implemented using `torch.nn.Module`) through its optimized backend. The examples in this repo demonstrate how `ORTModule` can be used to switch the training backend. 

## Examples

Outline the examples in the repository.

| Example                | Performance Comparison                      | Model Change                                |
|------------------------|---------------------------------------------|---------------------------------------------|
| HuggingFace BART       | See [BART](huggingface/BART.md)             | No model change required |
| HuggingFace BERT       | See [BERT](huggingface/BERT.md)             | No model change required |
| HuggingFace DeBERTa    | See [DeBERTa](huggingface/DeBERTa.md)       | See [this commit](https://github.com/microsoft/huggingface-transformers/commit/0b2532a4f1df90858472d1eb2ca3ac4eaea42af1)|
| HuggingFace DistilBERT | See [DistilBERT](huggingface/DistilBERT.md) | No model change required |
| HuggingFace GPT2       | See [GPT2](huggingface/GPT2.md)             | No model change required|
| HuggingFace RoBERTa    | See [RoBERTa](huggingface/RoBERTa.md)       | See [this commit](https://github.com/microsoft/huggingface-transformers/commit/b25c43e533c5cadbc4734cc3615563a2304c18a2)|
| t5-large               | See [T5](huggingface/T5.md)                 | See [this PR](https://github.com/microsoft/huggingface-transformers/pull/4/files) |
<!-- 
| `CONTRIBUTING.md` | Guidelines for contributing to the sample. |
-->


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
