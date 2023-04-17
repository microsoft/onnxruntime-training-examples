import argparse
import time
import numpy as np
import os

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import onnxruntime

def infer(args):
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

    # load model and update state...
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    model.load_state_dict(torch.load("pytorch_model.bin", map_location=torch.device(device)))
    model.eval()

    user_input = input("Question: ")
    while (user_input):
        print("user input:", user_input)
        # preprocess test data
        inputs = [(user_input, context)]

        # tokenize test data
        encoding = tokenizer.batch_encode_plus(inputs, padding=True, return_tensors="pt")
        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

        # if using onnnxruntime, convert to onnx format
        # ORT Python API Documentation: https://onnxruntime.ai/docs/api/python/api_summary.html
        if args.ort:
            # if not os.path.exists("model.onnx"):
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

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        model.to(device)

        # run inference
        start = time.time()
        if args.ort:
            # NOTE: this copies data from CPU to GPU
            # since our data is small, we are still faster than baseline pytorch
            # refer to ORT Python API Documentation for information on io_binding to explicitly move data to GPU ahead of time
            output = sess.run(None, ort_input)
        else:
            output = model(input_ids, attention_mask=attention_mask)
        end = time.time()

        # postprocess test data
        if args.ort:
            # ORT returns raw tensors
            max_start_logits = output[0][0].argmax()
            max_end_logits = output[1][0].argmax()
        else:
            # pytorch returns a dictionary of tensors
            max_start_logits = output.start_logits[0].argmax()
            max_end_logits = output.end_logits[0].argmax()
        ans_tokens = input_ids[0][max_start_logits: max_end_logits + 1]
        answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
        answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)
        print("Answer: ", answer_tokens_to_string)

        # brag about how fast we are
        print("Inference Time:", str(end - start), "seconds")

        user_input = input("Question: ")

def main():
    parser = argparse.ArgumentParser(description="DistilBERT Fine-Tuning")
    parser.add_argument("--ort", action="store_true", help="Use ORTModule")
    args = parser.parse_args()

    infer(args)

if __name__ == "__main__":
    main()
