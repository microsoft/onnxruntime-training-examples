## Using ACPT and ONNX Runtime to Speedup AzureML Online Endpoint Deployments

### Step 1: Register your model in your AML workspace

Command to generate whisper-small directory containing `.onnx` files (Note: this commad requires ONNX Runtime 1.15 or higher):
```bash
python -m onnxruntime.transformers.models.whisper.convert_to_onnx -m openai/whisper-small --output whisper-small --use_external_data_format
```
Alternative to use pretrained weights:
```bash
python -m onnxruntime.transformers.models.whisper.convert_to_onnx -m openai/whisper-small --output whisper-small --use_external_data_format --state_dict_path /path/to/pytorch_model.bin
```

In your AML Workspace, go to the "Models" tab. Register a new model from local files.
- Select the entire whisper-small directory
- Use all other defaults, register model

### Step 2: Register your environment in your AML workspace

In your AML Workspace, go to the "Environments" tab. Create a new environment.
- For "Select environment souce" select the "Create a new docker context"
- Add code from [Dockerfile](Dockerfile) to environment context
- Use all other defaults, deploy environment

### Step 3: Create Online Endpoint

In your AML Workspace, go to the "Endpoints" tab. Create a new endpoint.
- Select the model you previously registered
- Select the environment you previously registered
- Add [score.py](score.py) when asked to provide scoring file
- Set your compute target (ACPT requires >=64GB of disk storage)
- Use all other defaults, deploy endpoint

### Step 4: Test with [inference.py](inference.py)

Run `python inference.py` to test endpoint on mp3 data.
- Set url and api_key with relevant values from AML Workspace.
- For more details, see "Comsume" tab for your endpoint in your AML Workspace