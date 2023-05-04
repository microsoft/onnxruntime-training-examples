import joblib  
import numpy as np  
import os  
  
from inference_schema.schema_decorators import input_schema, output_schema  
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType  

import librosa
import numpy as np
from onnxruntime import InferenceSession
import os
import subprocess
import time
from transformers import WhisperProcessor  
  
# The init() method is called once, when the web service starts up.
def init():  
    global sess

    sess = InferenceSession("whisper-small_beamsearch.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
  
  
# The run() method is called each time a request is made to the scoring API.  
#  
# Shown here are the optional input_schema and output_schema decorators  
# from the inference-schema pip package. Using these decorators on your  
# run() method parses and validates the incoming payload against  
# the example input you provide here. This will also generate a Swagger  
# API document for your web service.  
# @input_schema('data', NumpyParameterType(np.array([[0.1, 1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0]])))  
# @output_schema(NumpyParameterType(np.array([4429.929236457418])))  
def run(data):  
    print(type(data))
    print(data)
    # Use the session object loaded by init().  
    N_FRAMES = 3000
    HOP_LENGTH = 160
    SAMPLE_RATE = 16000
    N_MELS = 80    
    min_length = 0
    max_length = 20
    repetition_penalty = 1.0
    beam_size = 1
    NUM_RETURN_SEQUENCES = 1
    input_shape = [1, N_MELS, N_FRAMES]

    audio = librosa.load(data)[0]

    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features

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
  
    # You can return any JSON-serializable object.  
    return transcription