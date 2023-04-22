import argparse
from azureml.core.run import Run
from dataclasses import dataclass
from datasets import Audio, DatasetDict, load_dataset
import evaluate
from pathlib import Path
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor
from typing import Any, Dict, List, Union

def init_nebula():
    import nebulaml as nm
    root_dir = Path(__file__).resolve().parent
    nebula_dir = root_dir / "nebula_checkpoints"
    nm.init(persistent_storage_path=str(nebula_dir)) # initialize Nebula

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="Hindi", task="transcribe")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    metric = evaluate.load("wer")
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

def finetune(args):
    if args.nebula:
        init_nebula()
        
    common_voice = DatasetDict()

    common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="train")
    common_voice["validation"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="validation")

    common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

    common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=4)

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    if args.ort_ds:
        from onnxruntime.training import ORTModule
        model = ORTModule(model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="output_dir",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        fp16=True,
        num_train_epochs=50,
        deepspeed="ds_config_zero_1.json" if args.ort_ds else None
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["validation"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(),
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor
    )

    train_result = trainer.train()

    # extract performance metrics
    train_metrics = train_result.metrics
    train_metrics["train_samples"] = len(common_voice["train"])
    trainer.log_metrics("train", train_metrics)

    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(common_voice["validation"])
    trainer.log_metrics("eval", eval_metrics)

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

def main():
    parser = argparse.ArgumentParser(description="Whisper Fine-Tuning")
    parser.add_argument("--ort_ds", action="store_true", help="Use ORTModule and DeepSpeed to accelerate training")
    parser.add_argument("--nebula", action="store_true", help="Enable nebula checkpointing")
    args = parser.parse_args()

    finetune(args)

main()