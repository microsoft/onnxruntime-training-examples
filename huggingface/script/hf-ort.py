import os
import shutil
import sys
import shlex
from datetime import datetime
import argparse

TRAINER_DIR = '../../huggingface-transformers/examples/pytorch'

MODEL_BATCHSIZE_DICT = {
    "bert-large" : '8',
    "distilbert-base" : '32',
    "gpt2" : '8',
    "bart-large" : '16',
    "t5-large" : '16',
    "deberta-v2-xxlarge" : '4',
    "roberta-large" : '16',
    "albert-base-v2": "1"
}

RUN_SCRIPT_DICT= {
    "bert-large" : ['run_mlm.py'],
    "distilbert-base" : ['run_mlm.py'],
    "gpt2" : ['run_clm.py'],
    "bart-large" : ['run_translation.py'],
    "t5-large" : ['run_translation.py'],
    "deberta-v2-xxlarge" : ['run_glue.py'],
    "roberta-large" : ['run_qa.py', 'trainer_qa.py', 'utils_qa.py'],
    "albert-base-v2" : ['run_qa.py', 'trainer_qa.py', 'utils_qa.py']
}

RUN_SCRIPT_DIR_DICT= {
    "bert-large" : 'language-modeling',
    "distilbert-base" : 'language-modeling',
    "gpt2" : 'language-modeling',
    "bart-large" : 'translation',
    "t5-large" : 'translation',
    "deberta-v2-xxlarge" : 'text-classification',
    "roberta-large" : 'question-answering',
    "albert-base-v2" : 'question-answering'
}

CONFIG_ARGS_DICT = {
    "pt-fp16" : [],
    "ort" : ['--ort'],
    "ds_s0" : ['--deepspeed', 'ds_config_zero_0.json'],
    "ds_s0_ort" : ['--ort', '--deepspeed', 'ds_config_zero_0.json'],
    "ds_s1" : ['--deepspeed', 'ds_config_zero_1.json'],
    "ds_s1_ort" : ['--ort', '--deepspeed', 'ds_config_zero_1.json']
}

# Check core SDK version number
print("The arguments are: " + str(sys.argv))

parser = argparse.ArgumentParser()

parser.add_argument("--gpu_cluster_name",
                        help="GPU cluster for the script to run on", type=str, required=False)

parser.add_argument("--hf_model",
                        help="Huggingface models to run", type=str, required=True,
                        choices=['bert-large', 'distilbert-base', 'gpt2', 'bart-large', 't5-large', 'deberta-v2-xxlarge', 'roberta-large', 'albert-base-v2'])

parser.add_argument("--run_config",
                        help="Run configuration indicating pytorch or ort, deepspeed stage", type=str, required=True,
                        choices=['pt-fp16', 'ort', 'ds_s0', 'ds_s0_ort', 'ds_s1', 'ds_s1_ort'])
# model params 
parser.add_argument("--model_batchsize",
                        help="Model batchsize per GPU", type=int, required=False)

parser.add_argument("--max_steps",
                        help="Max step that a model will run", type=int, default=8000, required=False)

parser.add_argument("--process_count",
                        help="Total number of GPUs (not GPUs per node)", type=int, required=False, default=8)

parser.add_argument("--node_count",
                        help="Node count", type=int, required=False, default=1)

parser.add_argument("--use_cu102",
                        help="Use Cuda 10.2 dockerfile. Default to False", action='store_true')

parser.add_argument("--local_run",
                        help="Run recipe locally, false for azureml run. Default to False", action='store_true')

args = parser.parse_args()                  

print(f"Running model: {args.hf_model}, config: {args.run_config} locally")

if args.model_batchsize:
    model_batchsize = args.model_batchsize
else:
    model_batchsize = MODEL_BATCHSIZE_DICT[args.hf_model]

base_args_dict = {
    "bert-large" : ['--model_name_or_path', 'bert-large-uncased', '--dataset_name', 'wikitext', '--dataset_config_name', 'wikitext-2-raw-v1', '--do_train', '--max_steps', args.max_steps, '--logging_steps', 200, '--output_dir', '/tmp/test-mlm-bbu', '--overwrite_output_dir', '--per_device_train_batch_size', model_batchsize, '--fp16'],
    "distilbert-base" : ['--model_name_or_path', 'distilbert-base-uncased', '--num_train_epochs', 1, '--dataset_name', 'wikitext', '--dataset_config_name', 'wikitext-2-raw-v1', '--do_train', '--max_steps', 4000, '--logging_steps', 200, '--output_dir', '/tmp/test-mlm-bbu', '--overwrite_output_dir', '--per_device_train_batch_size', model_batchsize, '--fp16'],
    "gpt2" : ['--model_name_or_path', 'gpt2', '--dataset_name', 'wikitext', '--num_train_epochs', 1, '--max_train_samples', 1000, '--dataset_config_name', 'wikitext-2-raw-v1', '--do_train', '--label_smoothing', 0.1, '--max_steps', 500, '--logging_steps', 200, '--overwrite_output_dir', '--output_dir', '/tmp/test-clm', '--per_device_train_batch_size', model_batchsize, '--fp16'],
    "bart-large" : ['--dataset_name', 'wmt16', '--dataset_config', 'ro-en', '--model_name_or_path', 'facebook/bart-large', '--output_dir', '/tmp/tst-translation', '--do_train', '--label_smoothing', 0.1, '--logging_steps', 200, '--overwrite_output_dir', '--per_device_train_batch_size', model_batchsize, '--predict_with_generate', '--source_lang', 'en', '--target_lang', 'ro', '--warmup_steps', 5, '--fp16', '--max_steps', args.max_steps],
    "t5-large" : ['--source_prefix', 'translate English to Romanian:', '--dataset_name', 'wmt16', '--dataset_config', 'ro-en', '--model_name_or_path', 't5-large', '--output_dir', '/tmp/tst-translation', '--do_train', '--label_smoothing', 0.1, '--logging_steps', 200, '--overwrite_output_dir', '--per_device_train_batch_size', model_batchsize, '--predict_with_generate', '--source_lang', 'en', '--target_lang', 'ro', '--warmup_steps', 5, '--fp16', '--max_steps', args.max_steps],
    "deberta-v2-xxlarge" : ['--model_name_or_path', 'microsoft/deberta-v2-xxlarge', '--task_name', 'MRPC', '--do_train', '--max_seq_length', 128, '--per_device_train_batch_size', model_batchsize, '--learning_rate', '3e-6', '--max_steps', args.max_steps, '--output_dir', '/tmp/deberta_res', '--overwrite_output_dir', '--logging_steps', 200, '--fp16'],
    "roberta-large" : ['--model_name_or_path', 'roberta-large', '--dataset_name', 'squad', '--do_train', '--per_device_train_batch_size', model_batchsize, '--learning_rate', '3e-5', '--max_steps', args.max_steps, '--max_seq_length', 384, '--doc_stride', 128, '--output_dir', '/tmp/roberta_res', '--overwrite_output_dir', '--logging_steps', 200, '--fp16'],
    "albert-base-v2" : ['--model_name_or_path', 'albert-base-v2', '--dataset_name', 'squad', '--do_train', '--per_device_train_batch_size', model_batchsize, '--learning_rate', '3e-5', '--max_steps', args.max_steps, '--max_seq_length', 384, '--doc_stride', 128, '--output_dir', '/tmp/alberta_res', '--overwrite_output_dir', '--logging_steps', 200, '--fp16']
}

if not args.local_run:
    if args.use_cu102:
        hf_ort_env = Environment.from_dockerfile(name='hf-ort-dockerfile-10.2', dockerfile='../docker/Dockerfile-10.2')
    else:
        hf_ort_env = Environment.from_dockerfile(name='hf-ort-dockerfile', dockerfile='../docker/Dockerfile')
    # This step builds a new docker image from dockerfile
    if not args.skip_docker_build:
        hf_ort_env.register(ws).build(ws).wait_for_completion()

model_experiment_name = 'hf-ortmodule-recipe-' + args.hf_model

model_run_args_base = base_args_dict[args.hf_model]
model_run_scripts = RUN_SCRIPT_DICT[args.hf_model]

# copy dependent run script to current folder
for script_file in model_run_scripts:
    model_run_script_path = os.path.normcase(os.path.join(TRAINER_DIR, RUN_SCRIPT_DIR_DICT[args.hf_model], script_file))
    shutil.copy(model_run_script_path, '.')

model_run_args_config = model_run_args_base + CONFIG_ARGS_DICT[args.run_config]
# use _xxberta.json for deberta and roberta
if args.hf_model in ['deberta-v2-xxlarge', 'roberta-large'] and args.run_config.startswith('ds_'):
    model_run_args_config[-1] = model_run_args_config[-1].replace('.json','_xxberta.json')
if args.hf_model in ['deberta-v2-xxlarge'] and not args.run_config.startswith('ds_') and args.process_count > 1:
    model_run_args_config += ['--sharded_ddp', 'simple']

#from subprocess import call, run
import sys
import subprocess
env = os.environ.copy()
if args.process_count == 1:
    env['CUDA_VISIBLE_DEVICES'] = '0'
    cmd_arry = [sys.executable, model_run_scripts[0]] + model_run_args_config
else:
    cmd_arry = [sys.executable, '-m', 'torch.distributed.launch', '--nproc_per_node', args.process_count, model_run_scripts[0]] + model_run_args_config
cmd_arry = [str(s) for s in cmd_arry]
cmd = ' '.join(cmd_arry)
subprocess.run(cmd_arry, env=env)
