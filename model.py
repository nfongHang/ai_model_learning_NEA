import numpy as np
import random

class Network:
    def __init__(self, sizes : list, weights=None, init_type="xa", biases=None) -> None:
        self.sizes=sizes
        self.layers=len(sizes)
        if weights==None:
            # No premade weights - choose what type of weight and bias initialisation
            match init_type:
                case "xa":
                    #xavier initialisation
                    self.weights = self.__xavier_init(sizes)
                case "he":
                    self.weights = self.__he_init(sizes)
        else:
            self.weights=weights
        
        if biases==None:
            self.biases=np.zeros(tuple(sizes[1:]))
        else:
            self.biases=biases
        
    def __xavier_init(self, sizes) -> list:
        rng = (6/(sizes[0]+sizes[-1]))**1/2
        return [[[random.uniform(-rng, rng) 
                 for w in range(self.sizes[l+1])] 
                 for n in range(self.sizes[l])] 
                 for l in range(self.layers-1)]


    def __he_init(self,sizes) -> list:
        return [[[np.random.normal(0, np.sqrt(2 / sizes[0]), (sizes[0], sizes[0])) 
                 for w in range(self.sizes[l+1])] 
                 for n in range(self.sizes[l])]
                 for l in range(self.layers-1)]

    def feed_forward(self, a) -> list:
        for l in range(self.layers):
            a = np.add([np.dot(np.transpose(self.weights[l])[n], a) 
                        for n in range(self.sizes[l])], self.biases[l]) 
        return a
    
    def mini_batch_SGD(self, epoches : int, mini_batch_size : int, train_data : list) -> None:
        """
        Mini Batch Stochastic Gradient Descent function
        Parameters:
         - epoches : number of epoches (how many runs of mini batches)
         - mini_batch_size : number of testdata to be used for training
         - train_data : array of tuples of training data and label
        """

        for epoch in range(epoches):
            self.mini_batch_update(train_data, mini_batch_size)
    
    def mini_batch_update(self, train_data, mini_batch_size):
        random.shuffle(train_data)
        nabla_w = np.zeros_like(self.weights)
        nabla_b = np.zeros_like(self.biases)
        for batch_num in range(mini_batch_size):
            mini_batch = train_data[batch_num*mini_batch_size:(batch_num+1)*mini_batch_size]
