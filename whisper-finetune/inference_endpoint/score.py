import numpy as np
from onnxruntime import InferenceSession
import os
from transformers import WhisperProcessor
import json 
import time

# Documentation: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints
# Troubleshooting: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-troubleshoot-online-endpoints
  
# The init() method is called once, when the web service starts up.
def init():  
    global SESS
    global PROCESSOR

    PROCESSOR = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")

    # The AZUREML_MODEL_DIR environment variable indicates  
    # a directory containing the model file you registered.  
    model_filename = "whisper-small/openai/whisper-small_beamsearch.onnx" 
    model_path = os.path.join(os.environ['AZUREML_MODEL_DIR'], model_filename)  

    SESS = InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
  
  
# The run() method is called each time a request is made to the scoring API.  
def run(data):
    json_data = json.loads(data)
    audio = np.array(json_data["audio"])
    
    n_frames = 3000
    sample_rate = 16000
    n_mels = 80    
    min_length = 0
    max_length = 20
    repetition_penalty = 1.0
    beam_size = 1
    num_return_sequences = 1
    input_shape = [1, n_mels, n_frames]

    inputs = PROCESSOR(audio, sampling_rate=sample_rate, return_tensors="pt")
    input_features = inputs.input_features

    ort_inputs = {
        "input_features": np.array(input_features, dtype=np.float32),
        "max_length": np.array([max_length], dtype=np.int32),
        "min_length": np.array([min_length], dtype=np.int32),
        "num_beams": np.array([beam_size], dtype=np.int32),
        "num_return_sequences": np.array([num_return_sequences], dtype=np.int32),
        "length_penalty": np.array([1.0], dtype=np.float32),
        "repetition_penalty": np.array([repetition_penalty], dtype=np.float32),
        "attention_mask": np.zeros(input_shape).astype(np.int32),
    }

    start_time = time.time()
    out = SESS.run(None, ort_inputs)[0]
    inference_time = time.time() - start_time
    transcription = PROCESSOR.batch_decode(out[0], skip_special_tokens=True)[0]
  
    # You can return any JSON-serializable object.  
    return {"transcription": transcription, "inference_time (seconds)": inference_time}

