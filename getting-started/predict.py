import numpy as np
import torch
import torchtext
import onnx
import onnxruntime as rt

basic_english_tokenizer=torchtext.data.utils.get_tokenizer("basic_english")

TEXT = torchtext.data.Field(tokenize=basic_english_tokenizer,
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)

train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)                            

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

example_sentence = "the"
tokenized_sentence = basic_english_tokenizer(example_sentence)

print(tokenized_sentence)

indexed_sentence = TEXT.numericalize([tokenized_sentence]).detach().numpy()

print(indexed_sentence)

session = rt.InferenceSession("transformer.onnx")

input_name = session.get_inputs()[0].name

run_options = rt.RunOptions()
run_options.only_execute_path_to_fetches = True
output = session.run(['output'], {input_name: indexed_sentence}, run_options)

print(len(output[0][0]))

# greedy decoder
def greedy_decoder(data):
	# index for largest probability each row
	return [np.argmax(s) for s in data]

decoded_output = greedy_decoder(output)

print(f'When I say: {example_sentence}, the next word is {decoded_output}')

# Map these back to words

output_sentence_words = [TEXT.vocab.itos[i] for i in decoded_output]
print(output_sentence_words)

# Setup a loop to sequence through the words



