import argparse
from datasets import Audio, load_dataset
import time
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

def infer(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    common_voice = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test")
    common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

    model = WhisperForConditionalGeneration.from_pretrained("output_dir/checkpoint-2000")
    model.eval()

    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")

    inputs = processor(common_voice["audio"][0]["array"], return_tensors="pt")
    input_features = inputs.input_features

    # if using onnnxruntime, convert to onnx format
    # ORT Python API Documentation: https://onnxruntime.ai/docs/api/python/api_summary.html
    if args.ort:
        import onnxruntime
        import os 
        import numpy as np
        
        if not os.path.exists("model.onnx"):
            torch.onnx.export(model, \
                            input_features, \
                            "model.onnx", \
                            input_names=["input_features"], \
                            output_names=["generated_ids"]) 

        sess = onnxruntime.InferenceSession("model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        ort_input = {
                "input_features": np.ascontiguousarray(input_features.numpy()),
            }
        
    input_features = input_features.to(device)
    model.to(device)

    # run inference
    print("running inference...")
    start = time.time()
    if args.ort:
        # NOTE: this copies data from CPU to GPU
        # since our data is small, we are still faster than baseline pytorch
        # refer to ORT Python API Documentation for information on io_binding to explicitly move data to GPU ahead of time
        generated_ids = sess.run(None, ort_input)
    else:
        generated_ids = model.generate(inputs=input_features)
    end = time.time()

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