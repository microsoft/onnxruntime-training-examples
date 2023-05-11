import argparse
from pathlib import Path

from azure.ai.ml import MLClient, command
from azure.identity import AzureCliCredential

def get_args(raw_args=None):
    parser = argparse.ArgumentParser(description="Whisper Finetune AML job submission")

    # workspace
    parser.add_argument(
        "--ws_config",
        type=str,
        default="ws_config.json",
        help="Workspace configuration json file with subscription id, resource group, and workspace name",
    )
    
    parser.add_argument("--compute", type=str, default="v100", help="Compute target to run job on")

    # accelerator hyperparameters
    parser.add_argument("--ort_ds", action="store_true", help="Use ORTModule and DeepSpeed to accelerate training")

    parser.add_argument("--torch_version", choices=["1.13", "2.0"], default="2.0", help="Specify PyTorch version")

    # parse args, extra_args used for job configuration
    args = parser.parse_args(raw_args)
    print(f"input parameters {vars(args)}")
    return args


def main(raw_args=None):
    args = get_args(raw_args)

    root_dir = Path(__file__).resolve().parent

    # connect to the workspace
    # documentation: https://learn.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml.mlclient?view=azure-python
    ws_config_path = root_dir / args.ws_config
    ml_client = MLClient.from_config(credential=AzureCliCredential(), path=ws_config_path)

    code_dir = root_dir / "finetune-code"

    # tags
    tags = vars(args)

    if args.ort_ds:
        command_str = "torchrun --nproc-per-node 8 run_speech_recognition_seq2seq_ort.py --model_name_or_path=openai/whisper-small --dataset_name=mozilla-foundation/common_voice_11_0 --dataset_config_name=hi --language=hindi --output_dir=output_dir --per_device_train_batch_size=8 --per_device_eval_batch_size=8 --num_train_epochs 100 --preprocessing_num_workers=16 --length_column_name=input_length --max_duration_in_seconds=30 --text_column_name=sentence --freeze_feature_encoder=False --group_by_length --fp16 --do_train --do_eval --predict_with_generate --optim adamw_ort_fused --deepspeed zero_stage_1.json"
    else:
        command_str = "torchrun --nproc-per-node 8 run_speech_recognition_seq2seq.py --model_name_or_path=openai/whisper-small --dataset_name=mozilla-foundation/common_voice_11_0 --dataset_config_name=hi --language=hindi --output_dir=output_dir --per_device_train_batch_size=8 --per_device_eval_batch_size=8 --num_train_epochs 100 --preprocessing_num_workers=16 --length_column_name=input_length --max_duration_in_seconds=30 --text_column_name=sentence --freeze_feature_encoder=False --group_by_length --fp16 --do_train --do_eval --predict_with_generate"

    # define the command
    # documentation: https://learn.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml.entities.command?view=azure-python
    command_job = command(
        description="ACPT Whisper Finetune Demo",
        display_name="whisper-finetune"+ ("-ort-ds" if args.ort_ds else ""),
        experiment_name="acpt-whisper-finetune-demo",
        code=code_dir,
        command=command_str,
        environment="acpt-whisper-torch{0}@latest".format(args.torch_version.replace(".", "")),
        compute=args.compute,
        instance_count=1,
        tags=tags,
        shm_size="16g",
    )

    # submit the command
    print("submitting job")
    returned_job = ml_client.jobs.create_or_update(command_job)
    print("submitted job")

    aml_url = returned_job.studio_url
    print("job link:", aml_url)


if __name__ == "__main__":
    main()
