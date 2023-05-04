import librosa
import numpy as np
from onnxruntime import InferenceSession
import os
import subprocess
import time
from transformers import WhisperProcessor

N_FRAMES = 3000
HOP_LENGTH = 160
SAMPLE_RATE = 16000
N_MELS = 80

def infer():
    min_length = 0
    max_length = 20
    repetition_penalty = 1.0

    audio = librosa.load("common_voice_hi_23795238.mp3")[0]

    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features

    # Documentation: https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/models/whisper/README.md
    if not os.path.exists("whisper-small"):
        subprocess.call(["python", "-m", "onnxruntime.transformers.models.whisper.convert_to_onnx", "-m", "openai/whisper-small", "--output", "whisper-small", "--use_external_data_format", "--state_dict_path", "pytorch_model.bin"])

    sess = InferenceSession("whisper-small/openai/whisper-small_beamsearch.onnx", providers=["CUDAExecutionProvider"])
    
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

def main():
    infer()

main()
