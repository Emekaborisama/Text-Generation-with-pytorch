import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import string
import unidecode
import random
import torch
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import torch
import torch.nn as nn
from torch.autograd import Variable

# Check if GPU is available
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
else: 
    print('No GPU available, training on CPU; consider making n_epochs very small.')

filename = './data/letter.txt'

def preprocess_load(filename):
  with open(filename) as f:
    textfile = f.read()

  train_data= textfile.lower().split()

  trigrams = [([train_data[i], train_data[i + 1]], train_data[i + 2])
            for i in range(len(train_data) - 2)]
  chunk_len=len(trigrams)


  #vocab
  vocab = set(train_data)
  voc_len=len(vocab)
  word_to_ix = {word: i for i, word in enumerate(vocab)}

  #convert to tensor
  inp=[]
  tar=[]
  for context, target in trigrams:
    context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
    inp.append(context_idxs)
    targ = torch.tensor([word_to_ix[target]], dtype=torch.long)
    tar.append(targ)
  return inp, tar, word_to_ix, vocab, voc_len, chunk_len

inp, tar, word_to_ix, vocab, voc_len, chunk_len= preprocess_load(filename = filename)



class GRUmodel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(GRUmodel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size*2, hidden_size, n_layers,batch_first=True,
                          bidirectional=False)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))





def train(inp, target):
    hidden = model.init_hidden().cuda()
    model.zero_grad()
    loss = 0
    
    for c in range(chunk_len):
        output, hidden = model(inp[c].cuda(), hidden)
        loss += criterion(output, target[c].cuda())

    loss.backward()
    model_optimizer.step()

    return loss.data.item() / chunk_len


n_epochs = 10
print_every = 100
plot_every = 10
hidden_size = 100
n_layers = 1
lr = 0.01
best_valid_loss = float('inf')

model = GRUmodel(voc_len, hidden_size, voc_len, n_layers)
model_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

all_losses = []
loss_avg = 0
if(train_on_gpu):
    model.cuda()
for epoch in range(1, n_epochs + 1):
    loss = train(inp,tar)       
    loss_avg += loss

    if epoch % print_every == 0:
        print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 50, loss))
        #print(evaluate('ge', 200), '\n')

    if epoch % plot_every == 0:
        all_losses.append(loss_avg / plot_every)
        loss_avg = 0
    if loss < best_valid_loss:
        torch.save(model, './model/save.pt')



    
