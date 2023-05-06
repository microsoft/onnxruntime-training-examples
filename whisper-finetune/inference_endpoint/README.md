## Using ACPT and ONNX Runtime to Speedup AzureML Online Endpoint Deployments

### Step 1: Register your model

In your AML Workspace, go to the "Models" tab. Register a new model from job output.
- Select a previous fine-tune job
- Select "Unspecified type" for model type 
- Select `model/pytorch_model.bin` for job output
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
