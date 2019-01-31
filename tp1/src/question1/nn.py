import numpy as np


class NN(object):
    def __init__(self, hidden_dims=(1024, 2048), n_hidden=2, mode='train', datapath=None, model_path=None):
        self.hidden_dims = hidden_dims
        self.n_hidden = n_hidden
        self.mode = mode
        self.datapath = datapath
        self.model_path = model_path
        self.weights = []

    def initialize_weights(self, weights):
        self.weights = weights

    def forward(self, input):
        x = input
        for weight in self.weights:
            x = self.activation(x @ weight)

    def activation(self, input):
        # RELU for now
        return np.max(0, input)

    def loss(self, prediction):
        pass

    def softmax(self, input):
        pass

    def backward(self, cache, labels):
        pass

    def update(self, grads):
        pass

    def train(self):
        pass

    def test(self):
        pass
