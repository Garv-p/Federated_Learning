import Aggregator
import client
import numpy as np



if __name__ == "__main__":
    model = None
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
        

 





