import argparse
from datasets import Audio, DatasetDict, load_dataset
import time
import torch
from transformers import AutoModelForSpeechSeq2Seq, WhisperProcessor
from transformers import GenerationMixin
from transformers import pipeline

from torch.utils.data import DataLoader
from typing import Any, Dict, List, Union
from dataclasses import dataclass
import numpy as np


processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")

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
    # common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

    # common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["test"], num_proc=4)

    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small")
    model.load_state_dict(torch.load("pytorch_model.bin", map_location=torch.device(device)), strict=False) # maybe try strict=False??
    # model.config.forced_decoder_ids = None
    # model.config.suppress_tokens = []
    model.eval()

    print(common_voice["test"][0])
    inputs = processor(common_voice["test"][0]["audio"]["array"], return_tensors="pt")
    input_features = inputs.input_features

    # if using onnnxruntime, convert to onnx format
    # ORT Python API Documentation: https://onnxruntime.ai/docs/api/python/api_summary.html
    if args.ort:
        if not os.path.exists("model.onnx"):
            torch.onnx.export(model, \
                            (input_ids, attention_mask), \
                            "model.onnx", \
                            input_names=["input_ids", "attention_mask"], \
                            output_names=["start_logits", "end_logits"]) 

        sess = onnxruntime.InferenceSession("model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        ort_input = {
                "input_ids": np.ascontiguousarray(input_ids.numpy()),
                "attention_mask" : np.ascontiguousarray(attention_mask.numpy()),
            }

    # if using onnnxruntime, convert to onnx format
    # ORT Python API Documentation: https://onnxruntime.ai/docs/api/python/api_summary.html
    if args.ort:
        from torch_ort import ORTModule
        model = ORTModule(model)

        # input_features = np.ascontiguousarray(input_features)

    input_features = input_features.to(device)
    model.to(device)

    # run inference
    print("running inference...")
    start = time.time()
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