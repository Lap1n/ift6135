import math

import numpy as np
import scipy
from scipy.special import softmax
from scipy.stats import entropy, stats

OUTPUT = 10


class NN(object):
    def __init__(self, input_dim=784, output_dim=10, hidden_dims=(1024, 2048), n_hidden=2, learning_rate=0.001,
                 mini_batch_size=100,
                 mode='train',
                 datapath=None,
                 model_path=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.hidden_dims = hidden_dims
        self.n_hidden = n_hidden
        self.mode = mode
        self.datapath = datapath
        self.model_path = model_path
        self.weights = []
        self.activation_output_cache = []
        self.h_cache = []
        self.x_cache = []

    def initialize_weights(self, type="normal"):
        self.weights = []
        input_dim = self.input_dim

        if type == "zero":
            for i in range(self.n_hidden):
                shape = (self.hidden_dims[i], input_dim)
                self.weights.append(np.zeros(shape))
                input_dim = self.hidden_dims[i]
            self.weights.append(np.zeros((self.output_dim, input_dim)))

        elif type == "normal":
            for i in range(self.n_hidden):
                shape = (self.hidden_dims[i], input_dim)
                self.weights.append(np.random.normal(0, 1, shape))
                input_dim = self.hidden_dims[i]
            self.weights.append(np.random.normal(0, 1, (self.output_dim, input_dim)))

        elif type == "glorot":
            d = math.sqrt(6 / (self.hidden_dims[-2] + self.hidden_dims[-1]))
            for i in range(self.n_hidden):
                shape = (self.hidden_dims[i], input_dim)
                self.weights.append(np.random.uniform(-d, d, shape))
                input_dim = self.hidden_dims[i]
            self.weights.append(np.random.uniform(-d, d, (self.output_dim, input_dim)))

    def forward(self, input):
        x = input
        for i in range(len(self.weights) - 1):
            self.x_cache.append(x)
            weight = self.weights[i]
            linear_output = x @ weight.transpose()
            self.h_cache.append(linear_output)
            x = self.activation(linear_output)
            self.activation_output_cache.append(x)
        last_weight_layer = self.weights[-1]
        return self.softmax(x @ last_weight_layer.transpose())

    @staticmethod
    def activation(input):
        # RELU for now
        return np.maximum(0, input)

    @staticmethod
    def loss(prediction, target, epsilon=1e-12):
        # TODO: use the cross entropy from scipy?
        # prediction = np.clip(prediction, epsilon, 1. - epsilon)
        # return -np.sum(np.sum(np.multiply(np.log(prediction), np.log(target)), axis=0)) / prediction.shape[0]
        return entropy(target) + entropy(target, prediction)

    @staticmethod
    def softmax(input):
        return softmax(input, axis=1)

    def backward(self, predictiction, labels):
        # first layer
        delta_w_cache = []
        d_loss_softmax = predictiction - labels
        d_loss_weight2 = np.multiply(d_loss_softmax, self.activation_output_cache[1])
        delta_w_cache.append(d_loss_weight2)
        d_loss_activation1 = np.multiply(d_loss_softmax, self.weights[-1])

        # second layer
        d_activation1_h1 = np.empty(self.h_cache[0].shape)
        d_activation1_h1[self.h_cache[0] <= 0] = 0
        d_loss_h1 = np.multiply(d_loss_activation1, d_activation1_h1)
        d_loss_w1 = np.multiply(d_loss_h1, self.x_cache[0])
        delta_w_cache.append(d_loss_w1)

        return delta_w_cache

    def update(self, grads):
        for i in range(len(grads)):
            avg_grads = np.mean(grads[i], axis=1)
            self.weights[i] = self.weights[i] - self.learning_rate * avg_grads

    def train(self, train_set, validation_set):
        x = train_set[0]
        y = train_set[1]

        for i in range(0, len(x), self.mini_batch_size):
            sample_x = x[i:i + self.mini_batch_size]
            sample_y = y[i:i + self.mini_batch_size]
            probabilities = self.forward(sample_x)
            prediction = np.argmax(probabilities, axis=1)
            loss = self.loss(prediction, sample_y)
            print("Loss : {}".format(loss))

            grads = self.backward(prediction, sample_y)
            self.update(grads)

    @staticmethod
    def accuracty(pred, y):
        y_pred = np.argmax(pred, axis=1)
        y = np.argmax(y, axis=1)
        total_score = 0
        for i in range(len(y_pred)):
            if y_pred == y:
                total_score += 1
        return total_score / len(y_pred) * 100

    def test(self, test_set):
        x = test_set[0]
        y = test_set[1]
        predictions = self.forward(x)
        loss = self.loss(predictions, y)
        print("Loss : {}".format(loss))
        accuracy = self.accuracty(predictions, y)
        print("Accuracy : {}".format(accuracy))
