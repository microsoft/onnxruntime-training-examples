# MobileVIT End-to-End Example for On-Device Training
This folder contains an end-to-end example for on-device training in a C# console app using the ONNXRuntime Training libraries.

On-device training contains two steps: an "offline" pre-processing step that happens off of the device in question, and the training step that happens on the device. 

## About the task
The task for this example is facial expression recognition, categorizing an image into one of the following emotions:
1. neutral
2. happy
3. sad
4. surprise
5. fear
6. disgust
7. angry

The huggingface MobileVIT xx-small model is fine-tuned on the huggingface DiffusionFER dataset to perform this classification.

## Contents
The `mobilevit_offline.ipynb` Python notebook creates the necessary training artifacts for training. The following generated files & folders should be **migrated or moved** from the "offline" device to the device to be used for training:
1. The training ONNX model (default name will be `training_model.onnx`)
2. The eval ONNX model (default name will be `eval_model.onnx`)
3. The optimizer ONNX model (default name will be `optimizer_model.onnx`)
4. The checkpoint files (default name of the folder will be `checkpoint`, and all files in this folder should be moved to the device to be used for on-device training)

The `mobilevit_console` folder contains the C# console app that performs the training & inferencing for this task.

Once the training step is finished, one more ONNX model file will be generated. Its default name will be `mobilevit_console/mobilevit_console/bin/Debug/net6.0/trained_mobilevit.onnx`.

## Prerequisites for offline processing step
Make sure PyTorch, transformers, and numpy are installed into your Python environment.

Check the [instructions for installing the onnxruntime-training Python package](https://onnxruntime.ai/) & install it.

## Prerequisites for on-device step
1. Using Git bash, [clone the DiffusionFER dataset](https://huggingface.co/datasets/FER-Universe/DiffusionFER) onto the device that will be doing the training:
```
git lfs install
git clone https://huggingface.co/kdhht2334/autotrain-diffusion-emotion-facial-expression-recognition-40429105176
```
    1. After downloading this dataset, you will have to unzip the folders for the different expressions.
2. Open the C# app in Visual Studio & use the NuGet Package Manager for the project to make sure that the `Microsoft.ML.OnnxRuntime.Managed`, `Microsoft.ML.OnnxRuntime.Training`, and `System.Drawing.Common` NuGet packages are installed. 
    1. If not, check the instructions [here](https://onnxruntime.ai/) and select Optimize Training -> Learning on the Edge -> Windows -> C# -> CPU, then follow the instructions on installation for the ONNX Runtime Training packages. 

## Running
The C# console app requires one command line argument that is the absolute path to the FER dataset. Optionally, any additional arguments it takes in should be the absolute path to images to be used for the inferencing stage.

The FER dataset path should point to a folder containing the following folders: `angry, disgust, fear, happy, neutral, sad, surprise`, and within each emotion folder should just be the PNG images (there should not be another folder within those folders).
