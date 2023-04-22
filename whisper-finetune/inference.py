import argparse
from datasets import Audio, DatasetDict, load_dataset
import time
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import GenerationMixin

from torch.utils.data import DataLoader
from typing import Any, Dict, List, Union
from dataclasses import dataclass

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
        forward_attention_mask (`bool`)
            Whether to return attention_mask.
    """

    processor: Any
    decoder_start_token_id: int
    forward_attention_mask: bool

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if self.forward_attention_mask:
            batch["attention_mask"] = torch.LongTensor([feature["attention_mask"] for feature in features])

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

def infer(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    common_voice = DatasetDict()

    common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test")
    common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

    common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["test"], num_proc=4)

    model = WhisperForConditionalGeneration.from_pretrained("output_dir/checkpoint-2000")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.eval()

    collate_fn = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, decoder_start_token_id=model.config.decoder_start_token_id, forward_attention_mask=False)
    dataloader = DataLoader(common_voice["test"], collate_fn=collate_fn, batch_size=4)

    if not args.ort:
        inputs = processor(common_voice["audio"][0]["array"], return_tensors="pt")
        input_features = inputs.input_features
    # decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id

    # if using onnnxruntime, convert to onnx format
    # ORT Python API Documentation: https://onnxruntime.ai/docs/api/python/api_summary.html
    if args.ort:
        from torch_ort import ORTModule
        model = ORTModule(model)

    # input_features = input_features.to(device)
    # decoder_input_ids = decoder_input_ids.to(device)
    model.to(device)

    # run inference
    print("running inference...")
    start = time.time()
    if args.ort:
        # NOTE: this copies data from CPU to GPU
        # since our data is small, we are still faster than baseline pytorch
        # refer to ORT Python API Documentation for information on io_binding to explicitly move data to GPU ahead of time
        for batch in dataloader:
            print(batch)
            decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
            decoder_input_ids = decoder_input_ids.to(device)
            input_features = batch["input_features"].to(device)
            generated_ids = model(input_features, decoder_input_ids=decoder_input_ids).logits 
        # print(output)
        # generated_ids = GenerationMixin.greedy_search(output)
    else:
        generated_ids = model.generate(inputs=input_features)
    end = time.time()

    print("generated_ids", generated_ids)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(transcription)

    # brag about how fast we are
    print("Inference Time:", str(end - start), "seconds")

def main():
    parser = argparse.ArgumentParser(description="Whisper Fine-Tuning")
    parser.add_argument("--ort", action="store_true", help="Use ORT Inference Session")
    args = parser.parse_args()

    infer(args)

main()