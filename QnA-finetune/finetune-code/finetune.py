import argparse
from pathlib import Path
import os

cloud_run = True
try:
    from azureml.core.run import Run
except ModuleNotFoundError:
    cloud_run = False


from datasets import load_dataset
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer, DefaultDataCollator

def init_nebula():
    import nebulaml as nm
    root_dir = Path(__file__).resolve().parent
    nebula_dir = root_dir / "nebula_checkpoints"
    nm.init(persistent_storage_path=str(nebula_dir)) # initialize Nebula

def get_args(raw_args=None):
    parser = argparse.ArgumentParser(description="DistilBERT Fine-Tuning")

    parser.add_argument("--ort", action="store_true", help="Use ORTModule")
    parser.add_argument("--deepspeed", action="store_true", help="Use deepspeed")
    parser.add_argument("--model_name", choices=["microsoft/deberta-base", "distilbert-base-uncased"], default="distilbert-base-uncased", help="Hugging Face Model ID")
    parser.add_argument("--nebula", action="store_true", help="Enable nebula checkpointing")

    args = parser.parse_args(raw_args)
    print(f"input parameters {vars(args)}")
    return args

def preprocess_function(examples, tokenizer=None):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# NOTE: Training pipeline adapted from https://huggingface.co/docs/transformers/tasks/question_answering
def main(raw_args=None):
    args = get_args(raw_args)

    if args.nebula:
        init_nebula()

    # load the SQuAD dataset from the Huggingface Datasets library
    squad = load_dataset("squad")

    # load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)

    if args.ort:
        from onnxruntime.training import ORTModule
        model = ORTModule(model)

    # tokenize the data
    tokenized_squad = squad.map(preprocess_function, fn_kwargs={"tokenizer": tokenizer}, batched=True)

    # initialize training arguments
    training_args_dict = {
        "output_dir": ".outputs", # for intermediary checkpoints
        "do_train": True,
        "do_eval": True,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "learning_rate": 2e-5,
        "num_train_epochs": 50,
        "weight_decay": 0.01,
        "fp16": True,
        "deepspeed": "ds_config_zero_1.json" if args.deepspeed else None,
        "torch_compile": True if (torch.__version__ >= "2.0.0" and not args.ort) else False,
        "label_names": ["start_positions", "end_positions"]
    }
    training_args = TrainingArguments(**training_args_dict)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_squad["train"],
        eval_dataset=tokenized_squad["validation"],
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=DefaultDataCollator(),
    )

    # train
    train_result = trainer.train()

    # extract performance metrics
    train_metrics = train_result.metrics
    train_metrics["train_samples"] = len(tokenized_squad["train"])
    trainer.log_metrics("train", train_metrics)

    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(tokenized_squad["validation"])
    trainer.log_metrics("eval", eval_metrics)

    if cloud_run:
        rank = os.environ.get("RANK", -1)
        if int(rank) == 0:
            # save trained model
            trained_model_folder = "model"
            trained_model_path = Path(trained_model_folder)
            trained_model_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(trained_model_path / "weights")

            # upload saved data to AML
            # documentation: https://learn.microsoft.com/en-us/python/api/azureml-core/azureml.core.run(class)?view=azure-ml-py
            run = Run.get_context()
            run.upload_folder(name="model", path=trained_model_folder)

if __name__ == "__main__":
    main()
