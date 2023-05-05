### Command to generate whisper-small directory with .onnx files. Note: this commad required ONNX Runtime 1.15 or higher.

```bash
python -m onnxruntime.transformers.models.whisper.convert_to_onnx -m openai/whisper-small --output whisper-small --use_external_data_format --state_dict_path pytorch_model.bin
```