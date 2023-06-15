import argparse
from azureml.core.run import Run

def get_args(raw_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    args = parser.parse_args(raw_args)
    return args

def main():
    args = get_args()
    # upload weights to AML
    # documentation: https://learn.microsoft.com/en-us/python/api/azureml-core/azureml.core.run(class)?view=azure-ml-py
    run = Run.get_context()
    run.upload_folder(name=output_dir, path=str(Path(output_dir)))

if __name__ == "__main__":
    main()
    