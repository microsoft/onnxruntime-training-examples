## On-Device Training

This directory includes tutorials for `On-Device Training` with ONNX Runtime. README file in each tutorial goes in depth on the task at hand and provides more context on how to setup for that tutorial.

> **Note**
> ONNX Runtime On-Device Training is under active development. Please refer to the getting started with [`Optimized Training`](https://onnxruntime.ai/index.html#getStartedTable) page for installation instructions. Please create an [issue](https://github.com/microsoft/onnxruntime-training-examples/issues/new) with your scenario and requirements if you encounter problems.

## What is On-Device Training?

`On-Device Training` refers to training a model on an edge device, without the data ever leaving the device. Such a form of training enables applications to leverage the user data without compromising their privacy.

On-Device Training can be used in scenarios like federated learning and model personalization.

## How to use ONNX Runtime for performing On-Device Training

The task of performing training on the device is be broken down into two phases.

- Offline phase: Preparation of prerequisite files for the actual training. This task is typically done on either the server or on the user's computer as an offline step. Files generated from this step will be consumed at training time on the device. The files generated include:

  - The training onnx model.
  - The eval onnx model.
  - The optimizer onnx model.
  - The checkpoint file.

  The onnx models and the checkpoint files can be generated with the help of ONNX Runtime's Python utility, [onnxblock](https://github.com/microsoft/onnxruntime/blob/main/orttraining/orttraining/python/training/onnxblock/README.md).

- Training phase: Once the prerequisite artifacts have been generated, they can be deployed to along with the application to perform training on the device. The training is done using one of [several language bindings](https://onnxruntime.ai/docs/install/#training-phase---on-device-training) currently supported by ONNX Runtime Training.

## What tutorials does this repo include?

- [Python notebook](python/mnist.ipynb): A Python Notebook that introduces `On-Device Training` by showcasing training with the MNIST dataset.
  Tags: `Python`, `MNIST`, `Getting-Started`, `Classification`
- [Android application](c-cpp/android/): Android application that shows how the ONNX Runtime C, C++ API can be used for training a model on an Android device.
  Tags: `Android`, `C, C++`, `Transfer-Learning`, `Classification`, `MobileNet`
- [C# console application](csharp/): C# application using MobileBERT and showcasing the Masked Language Modelling (MLM) task.
  Tags: `C#`, `MLM`, `MobileBERT`, `Language Models`

## How to learn more about ONNX Runtime On-Device Training?

You can learn more [here](https://onnxruntime.ai/docs/get-started/training-on-device.html).

Open an issue either on this repo [microsoft/onnxruntime-training-examples](https://github.com/microsoft/onnxruntime-training-examples) or on [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) with questions on `On-Device Training`.
