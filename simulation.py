import Aggregator
import client
import numpy as np
import torch
import torch.optim as optim
from transformers import BertForMaskedLM

import torch.nn as nn
import torch.nn.functional as F
import math


class Client:
    def __init__(self, id, data, model, batch_size, lr):
        self.id = id
        self.data = data
        self.model = model
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters())        

    def train(self, epochs, batch_size, lr):
        
        self.model.train()
    

        for epoch in range(epochs):    
            total_loss = 0.
            for batch in self.data:
                self.optimizer.zero_grad()
                data,targets = batch
                output = self.model.forward(data)
                loss = self.criterion(output.view(-1, self.model.ntokens), targets)
                #need to reread what backward and optimizer do here. fit into brain conceptually
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f'Epoch {epoch+1}, Loss: {total_loss/len(self.data)}')

        self.model.eval()
        torch.no_grad()
        for batch in self.data:
            data, targets = batch
            output = self.model.forward(data)
            loss = self.criterion(output.view(-1, self.model.ntokens), targets)
            total_loss += loss.item()



            

    def predict(self, x):
        #TODO
        self.model.eval()
        torch.no_grad()
        output = self.model.forward(x)
        return output


    def update_model(self, new_model):
        #TODO
        return None


    def get_model(self):
        return self.model
    
class Aggregator:
    def __init__self(self, clients, global_model):
        self.clients = clients
        self.global_model = global_model

    def aggregate(self):
        #TODO
        return self.global_model
    

    def evaluate(self):
        #TODO
        return None

class TransformerModel(nn.Module):
    def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output
    
if __name__ == "__main__":
    

    dataset = load_dataset("karpathy/tiny_shakespeare")
    train_data = dataset["train"]["text"]
    test_data = dataset["test"]["text"]
    #lazy since assumes that all characters appear in the first entry 
    chars = list(set(train_data[0]))
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a 

    train_data = torch.tensor([encode(x) for x in train_data], dtype=torch.long)
    test_data = torch.tensor([encode(x) for x in test_data], dtype=torch.long)

    ntokens = len(chars)
    #single point model(no federated learning)
    model = TransformerModel(ntokens, 200, 2, 200, 2)
    single_model = Client(0, train_data, model, 32, .01)
    

    """
    num_clients = 10
    clients = []
    for i in range(num_clients):
        clients.append(client.Client(i, None, model))

    aggregator = Aggregator(clients, model)

    dataset = None
    #split dataset into num_clients parts
    splits = np.array_split(dataset, num_clients)
    for i in range(num_clients):
        clients[i].data = splits[i]
    
    lr = .01
    epochs = 10
    batch_size = 32
    num_rounds = 10
    for i in range(num_rounds):
        for i in range(num_clients):
            clients[i].train(epochs, batch_size, lr)

        aggregator.aggregate()
        aggregator.evaluate()
        new_model  = aggregator.get_model()
        for i in range(num_clients):
            clients[i].update_model(new_model)
            """
        

 





