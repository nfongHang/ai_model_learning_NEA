import numpy as np
import random

class Network:
    def __init__(self, sizes : list, weights=None, init_type="xa", biases=None):
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
        
    def __xavier_init(self, sizes):
        range = (6/(sizes[0]+sizes[-1]))**1/2
        return [[[random.uniform(-range, range) 
                 for w in range (self.sizes[l+1])] 
                 for n in range (self.sizes[l])] 
                 for l in range(self.layers-1)]


    def __he_init(self,sizes):
        return [[[np.random.normal(0, np.sqrt(2 / sizes[0]), (sizes[0], sizes[0])) 
                 for w in range(self.sizes[l+1])] 
                 for n in range(self.sizes[l])]
                 for l in range(self.layers-1)]

    def feed_forward(self, a):
        for l in range(self.layers):
            a = np.add([np.dot(np.transpose(self.weights[l])[n], a) 
                        for n in range(self.sizes[l])], self.bias[l]) 
            
        