# Get started with ONNX Runtime Training

Train a PyTorch transformer model with ONNX Runtime.

This sample trains the model defined here: https://pytorch.org/tutorials/beginner/transformer_tutorial.html. This is a transformer model composed of PyTorch transformer components such as the PositionalEncoder, TransformerEncoder and word Embedding.

Trained using the Wikitext-2 dataset (from `torchtext.dataset`), the language model assigns a probability for the likelihood of a given word (or a sequence of words) to follow a sequence of words.

This sample uses the base docker image `mcr.microsoft.com/azureml/onnxruntime-training:0.1-rc2-openmpi4.0-cuda10.2-cudnn7.6-nccl2.7.6` to train the model.

The purpose of this sample is to demonstrate the concepts and usage of ONNX Runtime Training, rather than to show its performance capabilities. For more complex samples that do demonstrate improved training performance at scale, please refer to the other samples in this repo:

* [NVIDIA BERT pre-training](../nvidia-bert)
* [HuggingFace GPT-2 fine-tuning](../huggingface-gpt2)

All instructions in this sample assume you have cloned this repo and are located in the root directory of the getting-started sample.

```bash
git clone https://github.com/microsoft/onnxruntime-training-examples
cd getting-started
```

## Code walk through

The training code is based on the original PyTorch training code, which is provided for reference in the `train.py` file.

### Model

The PyTorch model is defined in `model.py`.

### Trainer class

See `train_ort.py` for the ONNX Runtime Training code. In its `train_step` method, the ORTTrainer encapsulates one pass (forward pass, loss calculation, backward pass, and optimizer weight adjustment) of the training loop.

Parameters and options for training are supplied to `ORTTrainer` when it is created:

```python
trainer = ORTTrainer(model,                       # model
                     loss_with_flat_output,       # loss function
                     model_description(),         # model description
                     "SGDOptimizer",              # optimizer name
                     None,                        # optimizer attributes
                     learning_rate_description(), # learning rate description
                     device,                      # device
                     _opset_version=12)           # opset version
```

### Model and Loss function input

The PyTorch model and loss function be supplied separately to the trainer, or combined into the one `torch.nn.Module` network. In this case, they are specified separately in order to follow the original training code as closely as possible.

In the loss function specified to the ORTTrainer, the output is flatted so that it can be passed directly to the loss calculation in `train_step`.

```python
def loss_with_flat_output(output, target):
    output = output.view(-1, ntokens)
    return criterion(output, target)
```

### Model description

In order to generate an ONNX graph, which is optimized for training, the inputs and outputs of the model need to be specified explicitly. This is done by defining the `model_description()` function.

```bash
def model_description():
    input_desc = IODescription('src', [bptt, batch_size], torch.float32)
    label_desc = IODescription('label', [bptt, batch_size, ntokens], torch.int64)
    loss_desc = IODescription('loss', [], torch.float32)
    output_desc = IODescription('output', [bptt, batch_size, ntokens], torch.float32)
    return ModelDescription([input_desc, label_desc], [loss_desc, output_desc])
```

Each input and output has a name, shape and type.

### Optimizer

The training optimizer is specified by supplying one of a fixed number of options, as a string:

* `SGDOptimizer`
* `LambOptimizer`
* `AdamOptimizer`

### Learning rate description

A learning rate description is also required, in the form of name, shape and type.

```bash
def learning_rate_description():
    return IODescription('Learning_Rate', [lr,], torch.float32)
```

## Run the sample

1. Pre-requisites

   * A machine with at least one CUDA-capable GPU
   * Docker

2. Build the docker image

    ```bash
    docker/build.sh docker
    ```

    This will build a docker image called `ort-training-getting-started:latest`.

3. Launch the training code

    ```bash
    docker/launch.sh "python train_ort.py"
    ```
