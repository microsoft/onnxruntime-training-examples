import json 
import numpy as np
from onnxruntime import InferenceSession
import os
import time
from transformers import AutoTokenizer

# Documentation: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints
# Troubleshooting: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-troubleshoot-online-endpoints
  
# The init() method is called once, when the web service starts up.
def init():  
    global SESS
    global TOKENIZER
    # The AZUREML_MODEL_DIR environment variable indicates  
    # a directory containing the model file you registered.  
    model_filename = "outputs_beam_search.onnx" 
    model_filename = os.path.join(os.environ['AZUREML_MODEL_DIR'], "onnx", model_filename)  
    print("Loading ONNX Runtime InferenceSession...")
    SESS = InferenceSession(model_filename, providers=["CPUExecutionProvider"])
    print("...done!")
    
    print("Loading Hugging Face tokenizer for t5-small...")
    TOKENIZER = AutoTokenizer.from_pretrained("t5-small")
    print("...done!")

    print("Completed init()")
    
 
  
# The run() method is called each time a request is made to the scoring API.  
def run(data):
    json_data = json.loads(data)
    input_data = json_data["inputs"]["article"]
    
    input_ids = TOKENIZER(str(input_data), return_tensors="pt").input_ids

    ort_inputs = {
        "input_ids": np.array(input_ids, dtype=np.int32),
        "max_length": np.array([512], dtype=np.int32),
        "min_length": np.array([0], dtype=np.int32),
        "num_beams": np.array([1], dtype=np.int32),
        "num_return_sequences": np.array([1], dtype=np.int32),
        "length_penalty": np.array([1.0], dtype=np.float32),
        "repetition_penalty": np.array([1.0], dtype=np.float32)
    }
    
    start_time = time.time()
    out = SESS.run(None, ort_inputs)[0][0] # 0th batch, 0th sample
    inference_time = time.time() - start_time

    summary = TOKENIZER.decode(out[0], skip_special_tokens=True)
      
    print("Input Article:", input_data[0])
    print("Output Summary:", summary)
    print("Inference Time (seconds):", inference_time)

    # You can return any JSON-serializable object.  
    return {"summary": summary, "inference time (seconds)": inference_time}

# if __name__ == '__main__':
#     # NOTE: You need to comment out model_filename = os.path.join(...) in init() for local testing
#     init()
#     payload = json.load(open("payload.json"))
#     payload = str.encode(json.dumps(payload))
#     _ = run(payload)
#     exit(0)