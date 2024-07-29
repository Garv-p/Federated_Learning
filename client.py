import numpy as np

class Client:
    def __init__(self, id, data, model):
        self.id = id
        self.data = data
        self.model = model

    def train(self, epochs, batch_size, lr):
        #TODO
        self.model.train()

        for epoch in range(epochs):
            

    def predict(self, x):
        #TODO

    def update_model(self, new_model):
        #TODO

    def get_model(self):
        return self.model


