# Run Instruction
1. Follow [Prerequisites and Setup](README.md#Prerequisites) steps in [README.md](README.md)
2. Launch training script
If config.json is in `huggingface/azureml`
```bash
cd huggingface/azureml
python hf-ort.py --gpu_cluster_name <gpu_cluster_name> --hf_model bart-large --run_config ort
```
### Alternatively, pass AzureML Workspace info through parameters
```bash
cd huggingface/azureml
python hf-ort.py --workspace_name <your_workspace_name> --resource_group 
<resource_group> --subscription_id <your_subscription_id> --gpu_cluster_name <gpu_cluster_name> --hf_model bart-large --run_config ort
```

# Performance Comparison
| Run configuration           | PyTorch | ORTModule | Gain  |
| -----------------           | ------- | --------- | ----- |
| fp16                        | 47.43   | 58.71     | 23.8% |
| fp16 with deepspeed stage 1 | 48.56   | 63.23     | 30.2% |
Number reflects samples/sec on above run with 8 gpus.

# Convergence
