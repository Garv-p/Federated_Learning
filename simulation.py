import Aggregator
import client
import numpy as np
import torch
import torch.optim as optim
import transformer

import torch.nn as nn
import torch.nn.functional as F
import math





#TODO
# encode the dataset into numbers/formatting. DONE
# write the tranformer class in pytorch : 
# write the model training data and the model evaluation data. turn into graphs. 
# write the aggregator classs. Look into Fedavg and other papers
# look at the results. Compare with LEAF.


class Client:
    def __init__(self, id, data,val_data, model, batch_size, lr):
        self.id = id
        self.data = data
        self.val_data = val_data
        self.model = model
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters())        

    def train(self, epochs, batch_size, lr):
        
        val_batches = [get_batches(self.val_data[i], batch_size, 256) for i in range(len(self.val_data))]
        batches = [get_batches(self.data[i], batch_size, 256) for i in range(len(self.data))]
        losses = []
        vals = []
        for epoch in range(epochs):           
            self.model.train() 
            total_loss = 0.
            for x,y in batches:
                x = torch.tensor(x, dtype=torch.long)
                y = torch.tensor(y, dtype=torch.long)
                self.optimizer.zero_grad()
                src_mask = generate_square_subsequent_mask(len(x))
                output = model.forward(x,src_mask)
                loss = self.criterion(output.view(-1, ntokens), y.view(-1) )
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            losses.append(total_loss/len(self.data))
            print(f'Epoch {epoch+1}, Loss: {total_loss/len(self.data)}')
            if True:
                self.model.eval()
                torch.no_grad()
                for x,y in val_batches:
                    x = torch.tensor(x, dtype=torch.long)
                    y = torch.tensor(y, dtype=torch.long)
                    src_mask = generate_square_subsequent_mask(len(x))
                    output = model.forward(x,src_mask)
                    loss = self.criterion(output.view(-1, ntokens), y.view(-1) )
                    total_loss += loss.item()
                vals.append(total_loss/len(self.data))
                print(f'Validation Loss: {total_loss/len(self.data)}')
  
        return losses, vals




            

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
        #fedavg
        #n_i is the same across all clients, so I can just average the weights by num clients. could implement n_i for sets of varying sizes;
        for 
                
    def get_model(self):
        return self.global_model
    

    def evaluate(self):
        #TODO
        return None



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
    model = transformer.TransformerModel(ntokens, 200, 2, 200, 2)
    single_model = Client(0, train_data, model, 32, .01)

    #divide training data into 5 parts
    #need to check if this is the correct way to split the data
    splits = np.array_split(train_data, 5)

    num_clients = 5
    clients = []
    for i in range(num_clients):
        clients.append(client.Client(i, splits[i], model))

    aggregator = Aggregator(clients, model)


    
    lr = .01
    epochs = 20
    batch_size = 32
    num_rounds = 10
    client_train_loss = {}
    for i in range(num_rounds):
        for i in range(num_clients):
            train_loss, val_loss = clients[i].train(epochs, batch_size, lr)
            client_train_loss[i] = (train_loss)

        aggregator.aggregate()
        aggregator.evaluate()
        new_model  = aggregator.get_model()
        for i in range(num_clients):
            clients[i].update_model(new_model)

        

 





