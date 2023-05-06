## Using ACPT and ONNX Runtime to Speedup AzureML Online Endpoint Deployments

### Step 1: Register your model

Command to generate whisper-small directory containing `.onnx` files (Note: this commad requires ONNX Runtime 1.15 or higher):
```bash
python -m onnxruntime.transformers.models.whisper.convert_to_onnx -m openai/whisper-small --output whisper-small --use_external_data_format
```
Alternative to use pretrained weights (see parent directory for instructions on how to fine-tune openai/whisper using AzureML, ACPT, and ORT Training in order to generate pytorch_model.bin):
```bash
python -m onnxruntime.transformers.models.whisper.convert_to_onnx -m openai/whisper-small --output whisper-small --use_external_data_format --state_dict_path /path/to/pytorch_model.bin
```

In your AML Workspace, go to the "Models" tab. Register a new model from local files.
- Select the entire whisper-small directory
- Use all other defaults, register model

### Step 2: Register your environment

In your AML Workspace, go to the "Environments" tab. Create a new environment.
- For "Select environment souce" select the "Create a new docker context"
- Add code from [Dockerfile](Dockerfile) to environment context
- Use all other defaults, deploy environment

### Step 3: Deploy online endpoint

In your AML Workspace, go to the "Endpoints" tab. Create a new endpoint.
- Select the model you previously registered
- Select the environment you previously registered
- Add [score.py](score.py) when asked to provide scoring file
- Set your compute target (ACPT requires >=64GB of disk storage)
- Use all other defaults, deploy endpoint

Save endpoint_url, endpoint_deployment_name, and api_key to endpoint_config.json
![image](https://user-images.githubusercontent.com/31260940/236587489-2e00d7a3-457a-425a-b492-fbb71711bd1b.png)

### Step 4: Test with [inference.py](inference.py)

Run `python inference.py` to test endpoint on mp3 data.
