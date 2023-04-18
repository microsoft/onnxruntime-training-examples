import argparse
import time
import numpy as np
import os

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import onnxruntime

def infer(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load tokenizer to preprocess data
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # load test data
    context = """
        ONNX Runtime is an open source project that is designed to accelerate machine learning 
        across a wide range of frameworks, operating systems, and hardware platforms. It enables 
        acceleration of machine learning inferencing across all of your deployment targets using 
        a single set of API. ONNX Runtime automatically parses through your model to identify 
        optimization opportunities and provides access to the best hardware acceleration available.
        ONNX Runtime also offers training acceleration, which incorporates innovations from Microsoft 
        Research and is proven across production workloads like Office 365, Bing and Visual Studio.
        At Microsoft, ONNX Runtime is used as the primary Machine Learning inferencing solution for 
        products groups. ONNX Runtime serves over 1 trillion daily inferences across over 150 production 
        models covering all task domains. ONNX Runtime abstracts custom accelerators and runtimes to 
        maximize their benefits across an ONNX model. To do this, ONNX Runtime partitions the ONNX model 
        graph into subgraphs that align with available custom accelerators and runtimes. When operators 
        are not supported by custom accelerators or runtimes, ONNX Runtime provides a default runtime 
        that is used as the fallback execution — ensuring that any model will run. 
        """

    # load model and update state...
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    model.load_state_dict(torch.load("pytorch_model.bin", map_location=torch.device(device)))
    model.eval()

    if args.ort:
        from onnxruntime.training import ORTModule
        model = ORTModule(model)

    question = input("Question: ")
    while (question):
        # tokenize test data
        encoding = tokenizer(question, context, return_tensors="pt")
        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        model.to(device)

        # run inference
        start = time.time()
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

        # reset data location for next question
        input_ids = input_ids.to("cpu")
        attention_mask = attention_mask.to("cpu")
        model.to("cpu")

        question = input("Question: ")

def main():
    parser = argparse.ArgumentParser(description="DistilBERT Fine-Tuning")
    parser.add_argument("--ort", action="store_true", help="Use ORTModule")
    args = parser.parse_args()

    infer(args)

if __name__ == "__main__":
    main()
