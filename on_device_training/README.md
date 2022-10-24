## On-Device Training

This directory includes tutorials for on-device training with onnxruntime. README file in each tutorial goes in depth on the task at hand and provides more context on how to setup for that tutorial.

This file answers some common questions around on-device training with onnxruntime.

## What is On-Device Training?
On-device training refers to training a model on the device, without the data ever leaving the device. Such a training enables applications to leverage the user data without compromising users privacy.

Learning on the edge (aka on-device training) can be used in scenarios like federated learning and model personalization.

## How to build onnxruntime for On-Device Training?
Build onnxruntime from source with the added flag `--enable_training --enable_training_on_device`. Example build command for windows build:
```sh
./build.bat --parallel --cmake_generator "Visual Studio 16 2019" --enable_training --enable_training_on_device --skip_tests

```
Refer to the README for each tutorial to learn more on how to build onnxruntime from source for that task.

## How to use onnxruntime for performing On-Device Training
The task of performing training on the device should be broken down into two steps.
- Offline preparation of prerequisite files for the actual training. This task is typically done on either the server or on the users computer as an offline step. Files generated from this step will be needed to perform the on-device training. The files generated include:
  - The training onnx model.
  - The eval onnx model.
  - The optimizer onnx model.
  - The checkpoint file.
  - The data for the model.

  The onnx models and the checkpoint files can be generated with the help of onnxruntime's python utility, [onnxblock](https://github.com/microsoft/onnxruntime/blob/main/orttraining/orttraining/python/training/onnxblock/README.md).
  The data for the models is considered part of the user's code and is left upto the user to prepare for their application.

- The built onnxruntime library and the offline files generated should be used while writing the training loop that will be executed on the device. User's should copy and include the C/C++ header files and onnxruntime library and invoke them as required for training.

## Where are the C/C++ APIs for On-Device Training?
- [C APIs](https://github.com/microsoft/onnxruntime/blob/main/orttraining/orttraining/training_api/include/onnxruntime_training_c_api.h)
- [C++ APIs](https://github.com/microsoft/onnxruntime/blob/main/orttraining/orttraining/training_api/include/onnxruntime_training_cxx_api.h)


## What tutorials does this repo include?
- [Android application for classification using cifar10](android_demo/README.md).

## How to learn more about onnxruntime On-Device Training?
Open an issue either on this repo [microsoft/onnxruntime-training-examples](https://github.com/microsoft/onnxruntime-training-examples) or on [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) with questions on on-device training.
