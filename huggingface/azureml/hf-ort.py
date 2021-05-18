import os
import requests
import sys
import shlex
from datetime import datetime
import argparse

# AzureML libraries
import azureml.core
from azureml.core import Experiment, Workspace, Datastore, Run, Environment
from azureml.core.compute import ComputeTarget, AmlCompute, AksCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import ScriptRunConfig
from azureml.core.runconfig import PyTorchConfiguration

# Check core SDK version number
print("SDK version:", azureml.core.VERSION)

print("The arguments are: " + str(sys.argv))

parser = argparse.ArgumentParser()

parser.add_argument("--workspace_name",
                        help="Name of the AzureML workspace", type=str, required=True)

parser.add_argument("--resource_group",
                        help="Resource group of the AzureML workspace", type=str, required=True)

parser.add_argument("--subscription_id",
                        help="Subscription of the AzureML workspace", type=str, required=True)

parser.add_argument("--gpu_cluster_name",
                        help="GPU cluster for the script to run on", type=str, required=True)

parser.add_argument("--min_cluster_node",
                        help="Min number of node for new cluster", type=str, required=True)

parser.add_argument("--max_cluster_node",
                        help="Max number of node for new cluster", type=str, required=True)

parser.add_argument("--hf_model",
                        help="Huggingface models to run. One of 'bert-large', 'distilbert-base', 'gpt2', 'bart-large', 't5-large'", type=str, required=True)

parser.add_argument("--run_config",
                        help="One of pt-fp16, ort, ds_s0, ds_s0_ort, ds_s1, ds_s1_ort", type=str, required=True)

parser.add_argument("--model_batchsize",
                        help="Model batchsize", type=str, required=False)

parser.add_argument("--process_count",
                        help="GPU process count", type=str, required=False, default=8)

parser.add_argument("--node_count",
                        help="node count", type=str, required=False, default=1)

args = parser.parse_args()                  

ws = Workspace.get(name=args.workspace_name, subscription_id=args.subscription_id, resource_group=args.resource_group)

# Create the compute cluster
gpu_cluster_name = args.gpu_cluster_name

# Verify that the cluster doesn't exist already
try:
    gpu_compute_target = ComputeTarget(workspace=ws, name=gpu_cluster_name)
    print('Found existing compute target.')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_ND40rs_v2', min_nodes=0, max_nodes=7)
    
    # create the cluster
    gpu_compute_target = ComputeTarget.create(ws, gpu_cluster_name, compute_config)
    gpu_compute_target.wait_for_completion(show_output=True)

hf_model = args.hf_model
run_config = args.run_config

if args.process_count > 1:
    torch_distributed_args = "python -m torch.distributed.launch --nproc_per_node 8 --use_env "
else:
    if args.run_config.contains("ds"):
        torch_distributed_args = "CUDA_VISIBLE_DEVICES=0 deepspeed"
    else:
        torch_distributed_args = "CUDA_VISIBLE_DEVICES=0 python"

model_batchsize_dict = {
        "bert-large" : '8',
        "distilbert-base" : '32',
        "gpt2" : '8',
        "bart-large" : '16',
        "t5-large" : '16'
    }

run_scripts_dict = {
    "bert-large" : 'run_mlm.py',
    "distilbert-base" : 'run_mlm.py',
    "gpt2" : 'run_clm.py',
    "bart-large" : 'run_translation.py',
    "t5-large" : 'run_translation.py'
}

base_args_dict = {
    "bert-large" : ['--model_name_or_path', 'bert-large-uncased', '--dataset_name', 'wikitext', '--dataset_config_name', 'wikitext-2-raw-v1', '--do_train', '--max_steps', 200, '--logging_steps', 200, '--output_dir', '/tmp/test-mlm-bbu', '--overwrite_output_dir', '--per_device_train_batch_size', 8, '--fp16'],
    "distilbert-base" : ['--model_name_or_path', 'distilbert-base-uncased', '--dataset_name', 'wikitext', '--dataset_config_name', 'wikitext-2-raw-v1', '--do_train', '--max_steps', 200, '--logging_steps', 200, '--output_dir', '/tmp/test-mlm-bbu', '--overwrite_output_dir', '--per_device_train_batch_size', 32, '--fp16'],
    "gpt2" : ['--model_name_or_path', 'gpt2', '--dataset_name', 'wikitext', '--dataset_config_name', 'wikitext-2-raw-v1', '--do_train', '--label_smoothing', 0.1, '--max_steps', 200, '--logging_steps', 200, '--overwrite_output_dir', '--output_dir', '/tmp/test-clm', '--per_device_train_batch_size', 8, '--fp16'],
    "bart-large" : ['--dataset_name', 'wmt16', '--dataset_config', 'ro-en', '--model_name_or_path', 'facebook/bart-large', '--output_dir', '/tmp/tst-translation', '--do_train', '--label_smoothing', 0.1, '--logging_steps', 200, '--overwrite_output_dir', '--per_device_train_batch_size', 16, '--predict_with_generate', '--source_lang', 'en', '--target_lang', 'ro', '--warmup_steps', 5, '--fp16', '--max_steps', 200],
    "t5-large" : ['--source_prefix', 'translate English to Romanian:', '--dataset_name', 'wmt16', '--dataset_config', 'ro-en', '--model_name_or_path', 't5-large', '--output_dir', '/tmp/tst-translation', '--do_train', '--label_smoothing', 0.1, '--logging_steps', 200, '--overwrite_output_dir', '--per_device_train_batch_size', 16, '--predict_with_generate', '--source_lang', 'en', '--target_lang', 'ro', '--warmup_steps', 5, '--fp16', '--max_steps', 200],
}

config_args_dict = {
    "pt-fp16" : [],
    "ort" : ['--ort'],
    "ds_s0" : ['--deepspeed', 'ds_config_zero_0.json'],
    "ds_s1" : ['--deepspeed', 'ds_config_zero_1.json'],
    "ds_s0_ort" : ['--ort', '--deepspeed', 'ds_config_zero_0.json'],
    "ds_s1_ort" : ['--ort', '--deepspeed', 'ds_config_zero_1.json'],
}

if args.model_batchsize:
    model_batchsize = args.model_batchsize
else:
    model_batchsize = model_batchsize_dict[hf_model]

hf_ort_env = Environment.from_dockerfile(name='hf-ort-dockerfile', dockerfile='../docker/Dockerfile')
hf_ort_env.register(ws).build(ws).wait_for_completion()

distr_config = PyTorchConfiguration(process_count=8, node_count=1)

model_experiment_name = 'hf-ortmodule-' + args.hf_model
model_run_args_base = base_args_dict[hf_model]
model_run_script = run_scripts_dict[hf_model]
# Create experiment for model
model_experiment = Experiment(ws, name=model_experiment_name)
model_run_args_config = model_run_args_base + config_args_dict[run_config]
# create script run config for the model+config
model_run_config = ScriptRunConfig(source_directory='../../huggingface-transformers/examples/language-modeling',
    script=model_run_script,
    arguments=model_run_args_config,
    compute_target=gpu_compute_target,
    environment=hf_ort_env,
    distributed_job_config=distr_config)

print("Submitting run for model: ", hf_model)
print("Submitting run for config: ", run_config)
run = model_experiment.submit(model_run_config)
run.add_properties({'model' : hf_model, 'config' : run_config, 'bs' : model_batchsize, 'gpus' : '8'})
