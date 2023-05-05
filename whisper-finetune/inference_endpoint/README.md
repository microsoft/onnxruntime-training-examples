#### Command to generate whisper-small directory containing `.onnx` files (Note: this commad requires ONNX Runtime 1.15 or higher):

```bash
python -m onnxruntime.transformers.models.whisper.convert_to_onnx -m openai/whisper-small --output whisper-small --use_external_data_format
```

#### Alternative to use pretrained weights:
```bash
python -m onnxruntime.transformers.models.whisper.convert_to_onnx -m openai/whisper-small --output whisper-small --use_external_data_format --state_dict_path /path/to/pytorch_model.bin
```
