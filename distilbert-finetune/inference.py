import argparse
import time
import numpy as np
import os

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import onnxruntime
from onnxruntime.capi import _pybind_state as C

def infer(args):
    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", device)
    
    # load tokenizer to preprocess data
    print("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # load test data
    context = """
        Beyonce Giselle Knowles-Carter (born September 4, 1981) is an American singer, songwriter, record producer and actress.
        Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame
        in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became
        one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously
        in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot
        100 number-one singles 'Crazy in Love' and 'Baby Boy'.
        """

    questions = ["When was Beyonce born?", 
                 "What areas did Beyonce compete in when she was growing up?", 
                 "When did Beyonce leave Destiny's Child and become a solo singer?", 
                 "What was the name of Beyonce's debut album?"]

    # preprocess test data
    inputs = []
    for question in questions:
        inputs.append((question, context))

    # tokenize test data
    print("tokenizing...")
    encoding = tokenizer.batch_encode_plus(inputs, padding=True, return_tensors="pt")
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    # load model and update state...
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    model.load_state_dict(torch.load("pytorch_model.bin", map_location=torch.device(device)))
    model.eval()

    # if using onnnxruntime, convert to onnx format
    # ORT Python API Documentation: https://onnxruntime.ai/docs/api/python/api_summary.html
    # Example Code for Inference: https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/notebooks/Inference_GPT2-OneStepSearch_OnnxRuntime_CPU.ipynb

    if args.ort:
        if not os.path.exists("model.onnx"):
            torch.onnx.export(model, \
                            (input_ids, attention_mask), \
                            "model.onnx", \
                            input_names=["input_ids", "attention_mask"], \
                            output_names=["start_logits", "end_logits"]) 

        sess = onnxruntime.InferenceSession("model.onnx", providers=C.get_available_providers())
        ort_input = {
                "input_ids": np.ascontiguousarray(input_ids.numpy()),
                "attention_mask" : np.ascontiguousarray(attention_mask.numpy()),
            }

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    model.to(device)

    # run inference
    print("running inference...")
    duration = []
    n_trials = 100
    for i in range(n_trials):
        start = time.time()
        if args.ort:
            # note: this copies data from CPU to GPU every time, yet we are still faster than baseline pytorch
            # refer to ORT Python API Documentation: https://onnxruntime.ai/docs/api/python/api_summary.html 
            #    for information on io_binding to explicitly move data to GPU ahead of time
            output = sess.run(None, ort_input)
        else:
            output = model(input_ids, attention_mask=attention_mask)
        duration.append(time.time() - start)
    average_inference_time = sum(duration) / n_trials

    # postprocess test data
    print("\n--------- RESULTS ---------")
    print("Context: ", context)
    for i in range(len(questions)):
        if args.ort:
            max_start_logits = output[0][i].argmax()
            max_end_logits = output[1][i].argmax()
        else:
            max_start_logits = output.start_logits[i].argmax()
            max_end_logits = output.end_logits[i].argmax()
        ans_tokens = input_ids[i][max_start_logits: max_end_logits + 1]
        answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
        answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)
        print("Question: ", questions[i])
        print("Answer: ", answer_tokens_to_string)

    # brag about how fast we are
    print("Average Inference Time:", average_inference_time, "seconds")

def main(raw_args=None):
    parser = argparse.ArgumentParser(description="DistilBERT Fine-Tuning")
    parser.add_argument("--ort", action="store_true", help="Use ORTModule")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU")
    args = parser.parse_args()

    infer(args)

if __name__ == "__main__":
    main()