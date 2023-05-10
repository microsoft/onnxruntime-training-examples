import argparse
from pathlib import Path
import json

from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment, BuildContext 
from azure.identity import AzureCliCredential

# run test on automode workspace
ws_config = json.load(open("ws_config.json"))
subscription_id = ws_config["subscription_id"]
resource_group = ws_config["resource_group"]
workspace_name = ws_config["workspace_name"]
compute = ws_config["compute"]
nproc_per_node = ws_config["nproc_per_node"]

model_configs = {
    "google/vit-base-patch16-224": {
        "pytorch_max_bs": 117,
        "ort_max_bs": 221
    },
    "apple/mobilevit-small": {
        "pytorch_max_bs": 104,
        "ort_max_bs": 124
    },
    "facebook/deit-base-patch16-224": {
        "pytorch_max_bs": 117,
        "ort_max_bs": 221
    },
    "microsoft/beit-base-patch16-224-pt22k-ft22k": {
        "pytorch_max_bs": 110,
        "ort_max_bs": 185
    },
    "microsoft/swinv2-base-patch4-window12-192-22k": {
        "pytorch_max_bs": 71,
        "ort_max_bs": 107
    }
}

def get_args(raw_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="google/vit-base-patch16-224", choices=list(model_configs.keys()).extend("all"), help="Hugging Face Model ID")
    parser.add_argument("--batch_size", default="max", choices=["max", "max+1", "pytorch_max", "8"], help="Per device batch size")

    args = parser.parse_args(raw_args)
    return args

def main(raw_args=None):
    args = get_args(raw_args)

    ml_client = MLClient(
        AzureCliCredential(), subscription_id, resource_group, workspace_name
    )

    root_dir = Path(__file__).resolve().parent
    environment_dir = root_dir / "environment"
    code_dir = root_dir / "finetune-code"
    experiment_name = "vision-ort-experiment"
    num_train_epochs = 100
    dataset = "DeepFashion_jsonl"
    dataloader_num_workers = 4

    if args.model_name == "all":
        models_to_run = model_configs.keys()
    else:
        models_to_run = [args.model_name]
    
    for model in models_to_run:
        if args.batch_size == "max":
            bs = model_configs[model]["pytorch_max_bs"]
        elif args.batch_size == "max+1":
            bs = model_configs[model]["pytorch_max_bs"] + 1
        elif args.batch_size == "pytorch_max":
            bs = model_configs[model]["pytorch_max_bs"]
        elif args.batch_size == "8":
            bs = 8

        pytorch_job = command(
            code=code_dir,  # local path where the code is stored
            command=f"torchrun --nproc_per_node={nproc_per_node} run_image_classification.py \
                        --model_name_or_path {model} \
                        --do_train --do_eval \
                        --train_dir {dataset}/train --validation_dir {dataset}/validation \
                        --fp16 True --num_train_epochs {num_train_epochs} \
                        --per_device_train_batch_size {bs} --per_device_eval_batch_size {bs} \
                        --remove_unused_columns False --ignore_mismatched_sizes True \
                        --output_dir output_dir --overwrite_output_dir --dataloader_num_workers {dataloader_num_workers}",
            environment=Environment(build=BuildContext(path=environment_dir)),
            experiment_name=experiment_name,
            compute=compute,
            display_name="pytorch-" + model,
            description=f"Train a vision DNN with PyTorch on the {dataset} dataset.",
            tags={"batch_size": str(bs)},
            shm_size="16g"
        )

        print("submitting PyTorch job for " + model)
        pytorch_returned_job = ml_client.create_or_update(pytorch_job)
        print("submitted job")

        pytorch_aml_url = pytorch_returned_job.studio_url
        print("job link:", pytorch_aml_url)

        if args.batch_size == "max":
            bs = model_configs[model]["ort_max_bs"]
        elif args.batch_size == "max+1":
            bs = model_configs[model]["ort_max_bs"] + 1
        elif args.batch_size == "pytorch_max":
            bs = model_configs[model]["pytorch_max_bs"]
        elif args.batch_size == "8":
            bs = 8

        ort_job = command(
            code=code_dir,  # local path where the code is stored
            command=f"torchrun --nproc_per_node={nproc_per_node} run_image_classification_ort.py \
                        --model_name_or_path {model} \
                        --do_train --do_eval \
                        --train_dir {dataset}/train --validation_dir {dataset}/validation \
                        --fp16 True --num_train_epochs {num_train_epochs} \
                        --per_device_train_batch_size {bs} --per_device_eval_batch_size {bs} \
                        --remove_unused_columns False --ignore_mismatched_sizes True \
                        --output_dir output_dir --overwrite_output_dir --dataloader_num_workers {2*dataloader_num_workers} \
                        --optim adamw_ort_fused --deepspeed zero_stage_1.json",
            environment=Environment(build=BuildContext(path=environment_dir)),
            environment_variables={"ORTMODULE_FALLBACK_POLICY": "FALLBACK_DISABLE"},
            experiment_name=experiment_name,
            compute=compute,
            display_name= "ort_ds-" + model,
            description=f"Train a vision DNN with ONNX Runtime on the {dataset} dataset.",
            tags={"batch_size": str(bs)},
            shm_size="16g"
        )

        print("submitting ORT job for " + model)
        ort_returned_job = ml_client.create_or_update(ort_job)
        print("submitted job")

        ort_aml_url = ort_returned_job.studio_url
        print("job link:", ort_aml_url)

if __name__ == "__main__":
    main()
