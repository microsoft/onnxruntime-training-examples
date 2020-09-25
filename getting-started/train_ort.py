import time
import math
import torch
import torch.nn as nn
import torchtext
from torchtext.data.utils import get_tokenizer
from model import TransformerModel
from onnxruntime.training import ORTTrainer, optim

basic_english_tokenizer=get_tokenizer("basic_english")

TEXT = torchtext.data.Field(tokenize=basic_english_tokenizer,
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)

train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bptt = 35
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def batchify(data, bsz):
    data = TEXT.numericalize([data.examples[0].text])    
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)

ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
emsize = 200 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value

model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

criterion = nn.CrossEntropyLoss()

def loss_with_flat_output(output, target):
    output = output.view(-1, ntokens)
    return criterion(output, target)
    
learning_rate = 0.1

def train():
    total_loss = 0.
    start_time = time.time()
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        loss, output = trainer.train_step(data, targets)

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.3f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, learning_rate,
                    elapsed * 1000 / log_interval,
                    loss, math.exp(loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source):
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            loss, outputs = trainer.eval_step(data, targets)
            total_loss += len(data) * loss.item()
    return total_loss / (len(data_source) - 1)

best_val_loss = float("inf")
epochs = 3 # The number of epochs
best_model = None

model_description = {'inputs':  [('src', ['bptt', 'batch_size']),
                                 ('label', ['bptt_x_batch_size'])],
                     'outputs': [('loss', [], True),
                                 ('output', ['bptt', 'batch_size', ntokens])]}

optimizer_config = optim.SGDConfig(lr=learning_rate)

trainer = ORTTrainer(model,                       # model
                     model_description,           # model description
                     optimizer_config,            # optimizer configuration
                     loss_with_flat_output)       # loss function

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model


test_loss = evaluate(best_model, test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)


