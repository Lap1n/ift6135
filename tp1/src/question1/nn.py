import math

import numpy as np
from scipy.special import softmax
from scipy.stats import entropy

OUTPUT = 10


class NN(object):
    def __init__(self, input_dim=784, output_dim=10, hidden_dims=(1024, 2048), n_hidden=2, learning_rate=0.001,
                 mini_batch_size=1000, num_epoch=10,
                 mode='train',
                 datapath=None,
                 model_path=None):
        self.num_epoch = num_epoch
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

    def forward(self, input, training_mode=True):
        x = input

        # TODO: need to find cleaner way to do this
        if training_mode is True:
            self.activation_output_cache.append(x)
        for i in range(len(self.weights) - 1):
            weight = self.weights[i]
            linear_output = x @ weight.transpose()
            x = self.activation(linear_output)
            if training_mode is True:
                self.h_cache.append(linear_output)
                self.activation_output_cache.append(x)
        last_weight_layer = self.weights[-1]
        return self.softmax(x @ last_weight_layer.transpose())

    @staticmethod
    def activation(input):
        # RELU for now
        return np.maximum(0, input)

    @staticmethod
    def loss(prediction, target, epsilon=1e-12):
        prediction = np.clip(prediction, epsilon, 1. - epsilon)
        loss_sum = 0
        for i in range(len(prediction)):
            loss_sum += entropy(target[i]) + entropy(target[i], prediction[i])
        return loss_sum

    @staticmethod
    def softmax(input):
        return softmax(input, axis=1)

    def backward(self, predictiction, labels):
        # # last layer
        # dl_dh3 = dl_ds * ds_dh3
        dl_dh3 = predictiction - labels
        # dl_dw3 = dl_dh3 * dh3_dw3 =   dl_dh3 * a2
        dl_dw3 = dl_dh3.transpose() @ self.activation_output_cache[2]
        # dl_da2 =  dl_dh3 * dh3_da2
        dl_da2 = dl_dh3 @ self.weights[2]

        # # second hidden layer
        # dl_dh2 = dl_da2 * da2_dh2
        dl_dh2 = dl_da2 * self.derivative_relu(self.h_cache[1])
        # dl_dw2 = dl_dh2 * da2_dw2
        dl_dw2 = dl_dh2.transpose() @ self.activation_output_cache[1]
        # dl_da1 = dl_dh2 * dh2_da1
        dl_da1 = dl_dh2 @ self.weights[1]

        # # first hidden layer
        # dl_dh1 = dl_da1 * da1_dh1
        dl_dh1 = dl_da1 * self.derivative_relu(self.h_cache[0])
        # dl_dw1 = dl_dh1 * dh2_dw1
        dl_dw1 = dl_dh1.transpose() @ self.activation_output_cache[0]

        dl_w_cache = [dl_dw1, dl_dw2, dl_dw3]

        self.reset_caches()
        return dl_w_cache

    def reset_caches(self):
        self.h_cache = []
        self.activation_output_cache = []

    def update(self, grads):
        for i in range(len(grads)):
            # TODO: NOT SURE IF I CAN DO THAT IF i WANT THE AVG
            avg_grads = grads[i] / self.mini_batch_size
            self.weights[i] = self.weights[i] - self.learning_rate * avg_grads

    def train(self, train_set, validation_set):
        x = train_set[0]
        y = train_set[1]
        for epoch_idx in range(self.num_epoch):
            print("[Epoch #{}]".format(epoch_idx + 1))
            for i in range(0, len(x), self.mini_batch_size):
                sample_x = x[i:i + self.mini_batch_size]
                sample_y = y[i:i + self.mini_batch_size]
                sample_y = self.convert_label_to_one_hot(sample_y)
                probabilities = self.forward(sample_x)
                loss = self.loss(probabilities, sample_y)
                print("Train loss : {}".format(loss))

                grads = self.backward(probabilities, sample_y)
                self.update(grads)
            self.test(validation_set)

    @staticmethod
    def convert_label_to_one_hot(labels):
        y_one_hots = np.zeros((labels.shape[0], 10))
        y_one_hots[np.arange(labels.shape[0]), labels] = 1
        return y_one_hots

    @staticmethod
    def accuracy(pred, y):
        y_pred = np.argmax(pred, axis=1)
        y = np.argmax(y, axis=1)
        total_score = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y[i]:
                total_score += 1
        return total_score / len(y_pred) * 100

    def evaluate(self, dataset, mode="Valid"):
        x = dataset[0]
        y = dataset[1]
        y = self.convert_label_to_one_hot(y)
        predictions = self.forward(x, training_mode=False)
        loss = self.loss(predictions, y)
        print("{} loss : {}".format(mode, loss))
        accuracy = self.accuracy(predictions, y)
        print("{} accuracy : {}".format(mode, accuracy))

    def test(self, test_set):
        self.evaluate(test_set, mode="Test")

    def validate(self, valid_set):
        self.evaluate(valid_set, mode="Valid")

    @staticmethod
    def derivative_relu(x):
        derivative = np.empty(x.shape)
        derivative[x <= 0] = 0
        derivative[x > 0] = 1
        return derivative
