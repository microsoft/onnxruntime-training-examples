# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from onnxruntime import InferenceSession

import argparse
import numpy as np
import time
import librosa

N_FRAMES = 3000
HOP_LENGTH = 160
SAMPLE_RATE = 16000
N_MELS = 80


def run_inference(args):
    min_length = args.min_length
    max_length = args.max_length
    repetition_penalty = args.repetition_penalty

    audio = librosa.load(args.input)[0]
    audio = np.expand_dims(audio[:30 * SAMPLE_RATE], axis=0)
    # expand audio to 30 seconds
    audio = np.tile(audio, (1, 30 * SAMPLE_RATE // audio.shape[1] + 1))[:, :30 * SAMPLE_RATE]

    from transformers import WhisperProcessor
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")
    inputs = processor(audio[0], return_tensors="pt")
    input_features = inputs.input_features

    sess = InferenceSession(args.model, opt, providers=["CUDAExecutionProvider"])
    beam_size = 1
    NUM_RETURN_SEQUENCES = 1
    input_shape = [1, N_MELS, N_FRAMES]

    ort_inputs = {
        "input_features": np.array(input_features, dtype=np.float32),
        "max_length": np.array([max_length], dtype=np.int32),
        "min_length": np.array([min_length], dtype=np.int32),
        "num_beams": np.array([beam_size], dtype=np.int32),
        "num_return_sequences": np.array([NUM_RETURN_SEQUENCES], dtype=np.int32),
        "length_penalty": np.array([1.0], dtype=np.float32),
        "repetition_penalty": np.array([repetition_penalty], dtype=np.float32),
        "attention_mask": np.zeros(input_shape).astype(np.int32),
    }

    out = sess.run(None, ort_inputs)[0]
    transcription = processor.batch_decode(out[0], skip_special_tokens=True)[0]
    print(transcription)

    # Timed run
    start = time.time()
    for i in range(10):
        _ = sess.run(None, ort_inputs)
    diff = time.time() - start
    print(f"time {diff/10} sec")


def get_args():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--input", default="./test.wav", help="input")
    parent_parser.add_argument("--max_length", type=int, default=20, help="default to 20")
    parent_parser.add_argument("--min_length", type=int, default=0, help="default to 0")
    parent_parser.add_argument("-b", "--num_beams", type=int, default=5, help="default to 5")
    parent_parser.add_argument("-bsz", "--batch_size", type=int, default=1, help="default to 1")
    parent_parser.add_argument("--repetition_penalty", type=float, default=1.0, help="default to 1.0")
    parent_parser.add_argument("--no_repeat_ngram_size", type=int, default=3, help="default to 3")

    required_args = parent_parser.add_argument_group("required input arguments")
    required_args.add_argument("--model", default="./whisper-model.onnx", help="model.")

    return parent_parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run_inference(args)
