import numpy as np
from types import FunctionType


tansig = lambda x: 2 / (1 + np.exp(-2*x)) - 1
dev_tansig = lambda x: 1 - tansig(x)**2
min_squares = lambda x: sum(map(lambda y: y**2, x)) / 2


class Neuron:

    def __init__(self, bias: float = 0, activation: FunctionType = tansig):
        self.bias: float = bias
        self.entries: np.ndarray[float] = np.array([])
        self.fuction: function = activation
        self.output: float
        self.error: float

    def set_entries(self, entries: np.ndarray[float]):
        self.entries = entries
    
    def calculate_output(self, weights: np.ndarray[float]):
        self.output = self.fuction(np.dot(self.entries, weights) + self.bias)
    
    def update_bias(self, learning_rate: float):
        self.bias += learning_rate * self.error

    def update_error(self, weights: np.ndarray[float], error: np.ndarray[float]):
        self.error = np.dot(weights, error)

    def set_error(self, error: float):
        self.error = error

    def __str__(self):
        return f"Neuron(bias={self.bias}, entries={self.entries}, error={self.error})"