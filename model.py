import numpy as np
import random

class Network:
    def __init__(self, sizes : list, weights=None, init_type="xa"):
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

    def __xavier_init(self, sizes):
        range = (6/(sizes[0]+sizes[-1]))**1/2
        return [[[random.uniform(-range, range) 
                 for w in range (self.sizes[l+1])] 
                 for n in range (self.sizes[l])] 
                 for l in range(self.layers-1)]


    def __he_init(self,sizes):
        return np.random.normal(0, np.sqrt(2 / sizes[0]), (sizes[0], sizes[0]))

