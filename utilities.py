import numpy as np

tansig = lambda x: 2 / (1 + np.exp(-2*x)) - 1

class Neuron:

    def __init__(self, bias: float = 0, activation:  function = tansig):
        self.bias: float = bias
        self.entries: np.ndarray[float] = np.array([])
        self.fuction: function = activation

    def add_entry(self, entry: float):
        self.entries = np.append(self.entries, entry)
    
    def get_output(self, weights: np.ndarray[float]):
        return self.fuction(np.dot(self.entries, weights) + self.bias)
    


