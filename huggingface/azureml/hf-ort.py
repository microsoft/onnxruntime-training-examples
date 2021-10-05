MODEL_BATCHSIZE_DICT = {
    "bert-large" : '14',
    "distilbert-base" : '60',
    "gpt2" : '10',
    "bart-large" : '24',
    "t5-large" : '16',
    "deberta-v2-xxlarge" : '4',
    "roberta-large" : '28'
}

RUN_SCRIPT_DICT= {
    "bert-large" : 'run_mlm.py',
    "distilbert-base" : 'run_mlm.py',
    "gpt2" : 'run_clm.py',
    "bart-large" : 'run_translation.py',
    "t5-large" : 'run_translation.py',
    "deberta-v2-xxlarge" : 'run_glue.py',
    "roberta-large" : 'run_qa.py'
}

RUN_SCRIPT_DIR_DICT= {
    "bert-large" : 'language-modeling',
    "distilbert-base" : 'language-modeling',
    "gpt2" : 'language-modeling',
    "bart-large" : 'translation',
    "t5-large" : 'translation',
    "deberta-v2-xxlarge" : 'text-classification',
    "roberta-large" : 'question-answering'
}

CONFIG_ARGS_DICT = {
    "pt-fp16" : [],
    "ort" : ['--ort'],
    "ds_s0" : ['--deepspeed', 'ds_config_zero_0.json'],
    "ds_s1" : ['--deepspeed', 'ds_config_zero_1.json'],
    "ds_s0_ort" : ['--ort', '--deepspeed', 'ds_config_zero_0.json'],
    "ds_s1_ort" : ['--ort', '--deepspeed', 'ds_config_zero_1.json'],
}

import os
import shutil
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

parser.add_argument("--gpu_cluster_name",
                        help="GPU cluster for the script to run on", type=str, required=True)

parser.add_argument("--hf_model",
                        help="Huggingface models to run", type=str, required=True,
                        choices=['bert-large', 'distilbert-base', 'gpt2', 'bart-large', 't5-large', 'deberta-v2-xxlarge', 'roberta-large'])

parser.add_argument("--run_config",
                        help="Run configuration indicating pytorch or ort, deepspeed stage", type=str, required=True,
                        choices=['pt-fp16', 'ort', 'ds_s0', 'ds_s0_ort', 'ds_s1', 'ds_s1_ort'])

parser.add_argument("--workspace_name",
                        help="Name of the AzureML workspace", type=str, required=False)

parser.add_argument("--resource_group",
                        help="Resource group that AzureML workspace belongs to", type=str, required=False)

parser.add_argument("--subscription_id",
                        help="Subscription of that AzureML workspace belongs to", type=str, required=False)
# model params 
parser.add_argument("--model_batchsize",
                        help="Model batchsize", type=int, required=False)

parser.add_argument("--max_steps",
                        help="Max step that a model will run", type=int, default=200, required=False)

parser.add_argument("--process_count",
                        help="Total number of GPUs (not GPUs per node)", type=int, required=False, default=8)

parser.add_argument("--node_count",
                        help="node count", type=int, required=False, default=1)

args = parser.parse_args()                  

azureml_run = (args.gpu_cluster_name != 'local')

if azureml_run:
    try:
        ws = Workspace.from_config()
    except:
        if args.workspace_name and args.subscription_id and args.resource_group:
            ws = Workspace.get(name=args.workspace_name, subscription_id=args.subscription_id, resource_group=args.resource_group)
        else:
            print("Please provide workspace name, subscription id and resource group")

    # Create the compute cluster
    gpu_cluster_name = args.gpu_cluster_name
    
    # Verify that the cluster doesn't exist already
    try:
        gpu_compute_target = ComputeTarget(workspace=ws, name=gpu_cluster_name)
        print('Found existing compute target.')
    except ComputeTargetException:
        print(f'Compute target not found. Please create a compute target by name {gpu_cluster_name}')
  
if args.model_batchsize:
    model_batchsize = args.model_batchsize
else:
    model_batchsize = MODEL_BATCHSIZE_DICT[args.hf_model]

base_args_dict = {
    "bert-large" : ['--model_name_or_path', 'bert-large-uncased', '--dataset_name', 'wikitext', '--dataset_config_name', 'wikitext-2-raw-v1', '--do_train', '--max_steps', args.max_steps, '--logging_steps', 1, '--output_dir', '/tmp/test-mlm-bbu', '--overwrite_output_dir', '--per_device_train_batch_size', model_batchsize, '--fp16', '--skip_memory_metrics'],
    "distilbert-base" : ['--model_name_or_path', 'distilbert-base-uncased', '--dataset_name', 'wikitext', '--dataset_config_name', 'wikitext-2-raw-v1', '--do_train', '--max_steps', args.max_steps, '--logging_steps', 1, '--output_dir', '/tmp/test-mlm-bbu', '--overwrite_output_dir', '--per_device_train_batch_size', model_batchsize, '--fp16', '--skip_memory_metrics'],
    "gpt2" : ['--model_name_or_path', 'gpt2', '--dataset_name', 'wikitext', '--dataset_config_name', 'wikitext-2-raw-v1', '--do_train', '--label_smoothing', 0.1, '--max_steps', args.max_steps, '--logging_steps', 1, '--overwrite_output_dir', '--output_dir', '/tmp/test-clm', '--per_device_train_batch_size', model_batchsize, '--fp16', '--skip_memory_metrics'],
    "bart-large" : ['--dataset_name', 'wmt16', '--dataset_config', 'ro-en', '--model_name_or_path', 'facebook/bart-large', '--output_dir', '/tmp/tst-translation', '--do_train', '--label_smoothing', 0.1, '--logging_steps', 1, '--overwrite_output_dir', '--per_device_train_batch_size', model_batchsize, '--predict_with_generate', '--source_lang', 'en', '--target_lang', 'ro', '--warmup_steps', 5, '--fp16', '--max_steps', args.max_steps, '--skip_memory_metrics'],
    "t5-large" : ['--source_prefix', 'translate English to Romanian:', '--dataset_name', 'wmt16', '--dataset_config', 'ro-en', '--model_name_or_path', 't5-large', '--output_dir', '/tmp/tst-translation', '--do_train', '--label_smoothing', 0.1, '--logging_steps', 1, '--overwrite_output_dir', '--per_device_train_batch_size', model_batchsize, '--predict_with_generate', '--source_lang', 'en', '--target_lang', 'ro', '--warmup_steps', 5, '--fp16', '--max_steps', args.max_steps, '--skip_memory_metrics'],
    # warning: hack microsoft/deberta-v2-xxlarge => microsoft/deberta-v2-xlarge
    "deberta-v2-xxlarge" : ['--model_name_or_path', 'microsoft/deberta-v2-xlarge', '--task_name', 'MRPC', '--do_train', '--max_seq_length', 128, '--per_device_train_batch_size', model_batchsize, '--learning_rate', '3e-6', '--max_steps', args.max_steps, '--output_dir', '/tmp/deberta_res', '--overwrite_output_dir', '--logging_steps', 1, '--fp16', '--skip_memory_metrics'],
    "roberta-large" : ['--model_name_or_path', 'roberta-large', '--dataset_name', 'squad', '--do_train', '--per_device_train_batch_size', model_batchsize, '--learning_rate', '3e-5', '--max_steps', args.max_steps, '--max_seq_length', 384, '--doc_stride', 128, '--output_dir', '/tmp/roberta_res', '--overwrite_output_dir', '--logging_steps', 1, '--fp16', '--skip_memory_metrics']
}

if azureml_run:
    hf_ort_env = Environment.from_dockerfile(name='hf-ort-dockerfile', dockerfile='../docker/Dockerfile')
    # This step builds a new docker image from dockerfile
    #hf_ort_env.register(ws).build(ws).wait_for_completion()
  
distr_config = PyTorchConfiguration(process_count=args.process_count, node_count=args.node_count)

model_run_args_base = base_args_dict[args.hf_model]
model_run_script = RUN_SCRIPT_DICT[args.hf_model]
model_run_script_path = os.path.normcase(os.path.join('/stage/huggingface-transformers/examples/pytorch', RUN_SCRIPT_DIR_DICT[args.hf_model], model_run_script))
model_run_args_config = model_run_args_base + CONFIG_ARGS_DICT[args.run_config]

if azureml_run:
  shutil.copy(model_run_script_path, '.')
  # Create experiment for model
  model_experiment_name = 'hf-ortmodule-recipe-' + args.hf_model
  model_experiment = Experiment(ws, name=model_experiment_name)
  # create script run config for the model+config
  model_run_config = ScriptRunConfig(source_directory='.',
      script=model_run_script,
      arguments=model_run_args_config,
      compute_target=gpu_compute_target,
      environment=hf_ort_env,
      distributed_job_config=distr_config)
  
  print(f"Submitting run for model: {args.hf_model}, config: {args.run_config}")
  run = model_experiment.submit(model_run_config)
  run.add_properties({'model' : args.hf_model, 'config' : args.run_config, 'bs' : model_batchsize, 'gpus' : str(args.process_count)})
else:
 import subprocess
 torch_launch_arguments = ['python', '-m', 'torch.distributed.launch', '--nproc_per_node', str(args.process_count)]
 model_script_arguments = torch_launch_arguments + [model_run_script_path] + [str(arg) for arg in model_run_args_config]
 print('')
 print(' '.join(model_script_arguments))
 print('')
 sys.stdout.flush()
 subprocess.run(model_script_arguments)
