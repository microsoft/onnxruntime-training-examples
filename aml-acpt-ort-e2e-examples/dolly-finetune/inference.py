import argparse
import time
import torch
from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

def infer(args):
    tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b")
    model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b")

    if args.ort:
        from onnxruntime.training import ORTModule
        model = ORTModule(model)
        # from torch_ort import ORTModule, DebugOptions, LogLevel
        # model = ORTModule(model, DebugOptions(save_onnx=False, onnx_prefix='dolly', log_level=LogLevel.VERBOSE))

    model.eval()

    generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

    example_question = "Explain to me the difference between nuclear fission and fusion."
    print("Question:", example_question)
    answer = generate_text(example_question)
    print("Answer:", answer[0]["generated_text"])

    user_input = input("Question: ")
    while (user_input):
        print("Question:", user_input)
        start_time = time.time()
        answer = generate_text(user_input)
        inference_time = time.time() - start_time
        print("Answer:", answer[0]["generated_text"])
        print("Inference Time:", inference_time, "seconds")

def main():
    parser = argparse.ArgumentParser(description="databricks/dolly-v2 inference demo")
    parser.add_argument("--ort", action="store_true", help="Use ORTModule")
    args = parser.parse_args()

    infer(args)

if __name__ == "__main__":
    main()
