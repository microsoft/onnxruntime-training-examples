# MobileBERT Demo

This directory contains an example of using the C# ORT training API. 

In order to generate the ONNX files and the training data, this directory uses torch.export & writes to files with the tokenized data, but any method to export an ONNX model can be used.

## Requirements

Check the installation instructions [here](https://onnxruntime.ai/) -- scroll down and select "Optimize Training," then select: "On-device Training" -> "Windows" -> "C#" -> "CPU" then follow the installation instructions provided.