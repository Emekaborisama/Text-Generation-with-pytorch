from flask import Flask,jsonify,request,render_template, make_response
from flask_cors import CORS, cross_origin
#from predict_model import text_generator
import os 
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
# Current file directory
here = os.path.dirname(__file__)

file_name = './data/letter.txt'

text_file = os.path.join(here, file_name)
with open(text_file) as f:
    text_filen = f.read()

def preprocess_load(textfile):

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



inp, tar, word_to_ix, vocab, voc_len, chunk_len= preprocess_load(textfile = text_filen)


PATH = './model/save.pt'
textpath = os.path.join(here, PATH)

model = torch.load(textpath, map_location=torch.device('cpu'))
model.eval()

def text_generator(prime_str, predict_len, temperature):
  hidden = model.init_hidden()

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




app = Flask(__name__)
cors = CORS(app)



@app.route("/")
def index():
    return("welcome to love letter generation pytorch model")


@app.route('/lovelettergen', methods = ['POST'])
def lovel():
    text = requests.get_json()
    text_g = text['text']
    result = text_generator(text_g, 40, temperature=3)
    return result

if __name__ == '__main__':
    app.run(host= '0.0.0.0',port = 8000, debug=True)

