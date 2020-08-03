import time
import math
import torch
import torch.nn as nn
import torchtext
from torchtext.data.utils import get_tokenizer
from model import TransformerModel
from onnxruntime.capi.ort_trainer import IODescription, ModelDescription, ORTTrainer

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
eval_batch_size = 20
train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)

ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
emsize = 200     # embedding dimension
nhid = 200       # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2      # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2        # the number of heads in the multiheadattention models
dropout = 0.2    # the dropout value

model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

criterion = nn.CrossEntropyLoss()

def loss_with_flat_output(output, target):
    output = output.view(-1, ntokens)
    return criterion(output, target)
    
lr = 0.001 # learning rate

def train():
    total_loss = 0.
    start_time = time.time()
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        if len(data) < bptt:
            print(f"len(data)={len(data)} < {bptt}")
            continue
        results = trainer.train_step(data, targets, torch.tensor([lr]))
        loss = results[0]

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:03.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, lr,
                    elapsed * 1000 / log_interval,
                    loss, math.exp(loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source):
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            if len(data) < bptt:
                print(f"len(data)={len(data)} < {bptt}")
                continue
            results = trainer.eval_step(data, targets)
            loss = results[0]
            total_loss += len(data) * loss.item()
    return total_loss / (len(data_source) - 1)

best_val_loss = float("inf")
epochs = 3 # The number of epochs
best_model = None

def model_description():
    input_desc = IODescription('src', [bptt, batch_size], torch.float32)
    label_desc = IODescription('label', [bptt, batch_size, ntokens], torch.int64)
    loss_desc = IODescription('loss', [], torch.float32)
    output_desc = IODescription('output', [bptt, batch_size, ntokens], torch.float32)
    return ModelDescription([input_desc, label_desc], [loss_desc, output_desc])

def learning_rate_description():
    return IODescription('Learning_Rate', [lr,], torch.float32)


trainer = ORTTrainer(model,                       # model
                     loss_with_flat_output,       # loss function
                     model_description(),         # model description
                     "SGDOptimizer",              # optimizer name
                     None,                        # optimizer attributes
                     learning_rate_description(), # learning rate description
                     device,                      # device
                     _opset_version=12)           # opset version


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


