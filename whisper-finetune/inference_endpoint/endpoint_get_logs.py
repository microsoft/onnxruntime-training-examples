import json
from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential

ws_config = json.load(open("ws_config.json"))
subscription_id = ws_config["subscription_id"]
resource_group = ws_config["resource_group"]
workspace_name = ws_config["workspace_name"]

ml_client = MLClient(
    AzureCliCredential(), subscription_id, resource_group, workspace_name
)

logs = ml_client.online_deployments.get_logs(
  name="whisper-onnx-1", endpoint_name="acpt-ort-whisper", lines=100, container_type="storage-initializer"
)

print(logs)
