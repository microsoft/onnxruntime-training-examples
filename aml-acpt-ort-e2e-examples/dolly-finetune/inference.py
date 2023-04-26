import argparse
import torch
from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

def infer(args):
    tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b")
    model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b")

    if args.ort:
        from onnxruntime.training import ORTModule
        model = ORTModule(model)

    model.eval()

    generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

    question = "Explain to me the difference between nuclear fission and fusion."
    print("Question:", question)
    answer = generate_text(question)
    print("Answer:", answer[0]["generated_text"])

def main():
    parser = argparse.ArgumentParser(description="databricks/dolly-v2 inference demo")
    parser.add_argument("--ort", action="store_true", help="Use ORTModule")
    args = parser.parse_args()

    infer(args)

if __name__ == "__main__":
    main()
