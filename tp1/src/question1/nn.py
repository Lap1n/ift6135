import math

import numpy as np
import scipy
from scipy.special import softmax
from scipy.stats import entropy, stats

INPUT_SIZE = 784
OUTPUT = 10


class NN(object):
    def __init__(self, hidden_dims=(1024, 2048), n_hidden=2, learning_rate=0.001, mini_batch_size=100, mode='train',
                 datapath=None,
                 model_path=None):
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
        if type == "zero":
            for i in range(self.n_hidden):
                shape = (self.hidden_dims[i],)

                self.weights.append(np.zeros(shape))
        elif type == "normal":
            for i in range(self.n_hidden):
                shape = (self.hidden_dims[i],)
                self.weights.append(np.random.normal(0, 1, shape))
        elif type == "glorot":
            d = math.sqrt(6 / (self.hidden_dims[-2] + self.hidden_dims[-1]))
            for i in range(self.n_hidden):
                shape = (self.hidden_dims[i], INPUT_SIZE)

                self.weights.append(np.random.uniform(-d, d, shape))

    def forward(self, input):
        x = input
        for i in range(len(self.weights) - 1):
            self.x_cache.append(x)
            weight = self.weights[i]
            linear_output = x @ weight
            self.h_cache.append(linear_output)
            x = self.activation(linear_output)
            self.activation_output_cache.append(x)
        last_weight_layer = self.weights[-1]
        return softmax(x @ last_weight_layer)

    @staticmethod
    def activation(input):
        # RELU for now
        return np.max(0, input)

    @staticmethod
    def loss(prediction, target, epsilon=1e-12):
        # TODO: use the cross entropy from scipy?
        # prediction = np.clip(prediction, epsilon, 1. - epsilon)
        # return -np.sum(np.sum(np.multiply(np.log(prediction), np.log(target)), axis=0)) / prediction.shape[0]
        return entropy(target) + entropy(target, prediction)

    @staticmethod
    def softmax(input):
        return softmax(input, axis=0)

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
        y = train_set[0]

        for i in range(0, len(x), self.mini_batch_size):
            sample_x = x[i:i + self.mini_batch_size]
            sample_y = y[i:i + self.mini_batch_size]
            prediction = self.forward(sample_x)
            loss = self.loss(prediction, sample_y)
            print("Loss : {}".format(loss))

            grads = self.backward(prediction, sample_y)
            self.update(grads)

    @staticmethod
    def accuracty(pred, y):
        y_pred = np.argmax(pred, axis=0)
        y = np.argmax(y, axis=0)
        total_score = 0
        for i in range(len(y_pred)):
            if y_pred == y:
                total_score += 1
        return total_score / len(y_pred) * 100

    def test(self, test_set):
        predictions = self.forward(x)
        loss = self.loss(predictions, y)
        print("Loss : {}".format(loss))
        accuracy = self.accuracty(predictions, y)
        print("Accuracy : {}".format(accuracy))
