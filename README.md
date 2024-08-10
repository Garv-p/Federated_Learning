# Federated Learning Crash Course
Implementeing and exploring the concept of federated learning by training transformers on the tiny shakespeare dataset. (also trying to see how transformers work)

Set up aggregator and client classes to simulate the process and training a simple tranformer based on Karpathys video and training on the tiny shakespeare dataset. Trying to recreate the paper [FedAvg with Fine Tuning](https://proceedings.neurips.cc/paper_files/paper/2022/file/449590dfd5789cc7043f85f8bb7afa47-Paper-Conference.pdf) to see how using the process described applies trasnformers and text generation. 


#TODO
- plot the loss cruves and examine how the model compares with LEAF benchmarks(will need to change current model structure somewhat.)
- implement homomorphic encryption(if thats not possible then just use a library) to learn
