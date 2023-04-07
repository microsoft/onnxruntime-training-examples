# MobileBERT Demo

This directory contains an example of using the C# ORT training API. 

In order to generate the ONNX files and the training data, this directory uses torch.export & writes to files with the tokenized data, but any method to export an ONNX model can be used.

## Requirements

The C# app requires the Microsoft.ML.OnnxRuntime.Training package, which must be built from source. Instructions on how to build the training package can be found [here](https://github.com/microsoft/onnxruntime/tree/main/csharp). Make sure to use the `--enable_training_apis` flag if building using a build script. 