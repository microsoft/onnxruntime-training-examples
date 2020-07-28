import numpy as np
import torch
import torchtext
import onnx
from model import TransformerModel

basic_english_tokenizer=torchtext.data.utils.get_tokenizer("basic_english")

TEXT = torchtext.data.Field(tokenize=basic_english_tokenizer,
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)

train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(train_txt.examples[0].text[0:100])

data = TEXT.numericalize([train_txt.examples[0].text])

print(data)

# greedy decoder
def greedy_decoder(data):
	# index for largest probability each row
	return [np.argmax(s) for s in data]

example_sentence = "the"
tokenized_sentence = basic_english_tokenizer(example_sentence)
print(tokenized_sentence)
indexed_sentence = TEXT.numericalize([tokenized_sentence])
print(indexed_sentence)

ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
print(ntokens)
emsize = 200  # embedding dimension
nhid = 200    # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2   # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2     # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value

model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

# Check that the model can be exported to TorchScript
scripted_module = torch.jit.script(model)
print(type(scripted_module))  # torch.jit.ScriptFuncion
print(scripted_module.code)

model.load_state_dict(torch.load('model.pt'))
model.eval()

output = model(indexed_sentence).detach().numpy()
print(output[0])

decoded_output = greedy_decoder(output)

print(decoded_output)

# Map these back to words

output_sentence_words = [TEXT.vocab.itos[i] for i in decoded_output]
print(output_sentence_words)

# Setup a loop to sequence through the words



