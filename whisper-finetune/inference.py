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
import librosa

def infer(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    audio = librosa.load("common_voice_hi_23795238.mp3")[0]

    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small")
    model.load_state_dict(torch.load("pytorch_model.bin", map_location=torch.device(device)), strict=False) # maybe try strict=False??
    model.eval()

    from transformers import WhisperProcessor
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")
    inputs = processor(audio, return_tensors="pt")
    input_features = inputs.input_features

    input_features = input_features.to(device)
    model.to(device)

    # run inference
    generated_ids = model.generate(inputs=input_features)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(transcription)

    # Timed run
    start = time.time()
    for i in range(10):
        _ = model.generate(inputs=input_features)
    diff = time.time() - start
    print(f"time {diff/10} sec")


def main():
    parser = argparse.ArgumentParser(description="Whisper Fine-Tuning")
    parser.add_argument("--ort", action="store_true", help="Use ORT Inference Session")
    args = parser.parse_args()

    infer(args)

main()