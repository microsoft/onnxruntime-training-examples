# Accelerate GPT2 fine-tuning with ONNX Runtime Training

This example uses ONNX Runtime Training to fine-tune the GPT2 PyTorch model maintained at https://github.com/huggingface/transformers.

You can run the training in Azure Machine Learning or in other environments.

## Setup

1. Clone this repo

    ```bash
    git clone https://github.com/microsoft/onnxruntime-training-examples.git
    cd onnxruntime-training-examples/DeepSpeedExamples
    ```

3. Update with ORT changes

    ```bash
    git apply ../deepspeed/Megatron-LM-v1.1.5-ZeRO3/deepspeed_megatron.patch
    ```

4. Sample scripts launching distributed Megatron-LM GPT2 runs:
   TBD
