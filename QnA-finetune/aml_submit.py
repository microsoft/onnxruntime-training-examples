import argparse
from pathlib import Path

from azure.ai.ml import MLClient, command
from azure.identity import AzureCliCredential

def run_config_to_args(run_config):
    mapping = {
        "no_acc": [],
        "ds": ["--deepspeed"],
        "ort": ["--ort"],
        "ds_ort": ["--deepspeed", "--ort"],
    }
    return mapping[run_config]

def get_args(raw_args=None):
    parser = argparse.ArgumentParser(description="QnA Finetune AML job submission")

    # workspace
    parser.add_argument(
        "--ws_config",
        type=str,
        required=True,
        help="Workspace configuration json file with subscription id, resource group, and workspace name",
    )
    
    parser.add_argument("--compute", type=str, required=True, help="Compute target to run job on")

    # accelerator hyperparameters
    parser.add_argument(
        "--run_config", choices=["no_acc", "ort", "ds", "ds_ort"], default="no_acc", help="Configs to run for model"
    )

    parser.add_argument("--torch_version", choices=["1.13", "2.0"], default="1.13", help="Specify PyTorch version")

    parser.add_argument("--model_name", choices=["microsoft/deberta-base", "distilbert-base-uncased"], default="distilbert-base-uncased", help="Hugging Face Model ID")

    parser.add_argument("--nebula", action="store_true", help="Enable nebula checkpointing")

    # parse args, extra_args used for job configuration
    args = parser.parse_args(raw_args)
    print(f"input parameters {vars(args)}")
    return args


def main(raw_args=None):
    args = get_args(raw_args)
    run_config_args = run_config_to_args(args.run_config)

    root_dir = Path(__file__).resolve().parent

    # connect to the workspace
    # documentation: https://learn.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml.mlclient?view=azure-python
    ws_config_path = root_dir / args.ws_config
    ml_client = MLClient.from_config(credential=AzureCliCredential(), path=ws_config_path)

    code_dir = root_dir / "finetune-code"

    # tags
    tags = vars(args)

    # define the command
    # documentation: https://learn.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml.entities.command?view=azure-python
    command_job = command(
        description="ACPT QnA Finetune Demo",
        display_name=f"{args.model_name}-finetune-{args.run_config}",
        experiment_name="acpt-QnA-finetune-demo",
        code=code_dir,
        command=(
            "python finetune.py" + 
            f" {' '.join(run_config_args)}" + 
            f" --model_name {args.model_name}" +
            (" --nebula" if args.nebula else "")
        ),
        environment="acpt-distilbert-torch{0}@latest".format(args.torch_version.replace(".", "")),
        distribution={
            "type": "pytorch",
            "process_count_per_instance": 8,
        },
        compute=args.compute,
        instance_count=1,
        tags=tags,
    )

    # submit the command
    print("submitting job")
    returned_job = ml_client.jobs.create_or_update(command_job)
    print("submitted job")

    aml_url = returned_job.studio_url
    print("job link:", aml_url)


if __name__ == "__main__":
    main()
