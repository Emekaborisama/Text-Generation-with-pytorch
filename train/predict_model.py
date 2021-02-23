import torch
import torch.nn as nn
from torch.autograd import Variable
from trainpytorch import GRUmodel, preprocess_load


import os 
import sys


filename = '.data/letter.txt'
inp, tar, word_to_ix, vocab, voc_len, chunk_len= preprocess_load(filename = filename)



PATH = './model/save.pt'
model = torch.load(PATH)
model.eval()

def text_generator(prime_str, predict_len, temperature):
  hidden = model.init_hidden().cuda()

  for p in range(predict_len):
    prime_input = torch.tensor([word_to_ix[w] for w in prime_str.split()], dtype=torch.long).cuda()
    inp = prime_input[-2:] #last two words as input
    output, hidden = model(inp, hidden)
        
    # Sample from the network as a multinomial distribution
    output_dist = output.data.view(-1).div(temperature).exp()
    top_i = torch.multinomial(output_dist, 1)[0]
        
    # Add predicted word to string and use as next input
    predicted_word = list(word_to_ix.keys())[list(word_to_ix.values()).index(top_i)]
    prime_str += " " + predicted_word
    #inp = torch.tensor(word_to_ix[predicted_word], dtype=torch.long)

  return prime_str

