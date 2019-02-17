from itertools import product

from question1.get_dataset import get_dataset
from question1.nn import NN
import numpy as np
import matplotlib.pyplot as plt

p = 10


def gradients_analysis(nn, train_set):
    x = train_set[0]
    y = train_set[1]

    max_differences = []

    n_list = []
    sample_x = x[0].reshape(1, -1)
    sample_y = y[0].reshape(1, -1)
    sample_y = nn.convert_label_to_one_hot(sample_y)
    pred, cache = nn.forward(sample_x)

    grads = nn.backward(cache, pred, sample_y)

    real_loss_derivate = grads[2][0][0:10]

    for i in range(0, 3):
        for k in range(1, 6, 1):
            n = k * 10 ** i
            n_list.append(n)

            epsilon = 1 / n

            losses_plus_epsilon = finite_loss(nn, sample_x, sample_y, epsilon)
            losses_minus_epsilon = finite_loss(nn, sample_x, sample_y, -epsilon)

            gradient_finite_difference = (losses_plus_epsilon - losses_minus_epsilon) / (2 * epsilon)

            max_difference = np.max(np.abs(gradient_finite_difference - real_loss_derivate))
            max_differences.append(max_difference)
    plt.plot(n_list, max_differences)
    plt.title("Maximum difference between true gradient and estimate")
    plt.xlabel("N")
    plt.ylabel("Maximum difference")
    plt.savefig("./results/difference.png")
    plt.show()


def finite_loss(nn, sample_x, sample_y, epsilon):
    approximations = np.empty(p)
    old_weights = nn.weights[2].copy()
    for i in range(0, p):
        nn.weights[2][0][i] += epsilon
        pred_prob, cache = nn.forward(sample_x)
        loss_epsilon = nn.loss(pred_prob, sample_y)
        approximations[i] = loss_epsilon
        nn.weights[2] = old_weights

    return approximations


def finite_difference_gradients():
    nn = NN(learning_rate=0.1, hidden_dims=(1024, 512), mini_batch_size=100, num_epoch=2)
    nn.initialize_weights("glorot")
    dataset = get_dataset()
    train = dataset[0]
    valid = dataset[1]

    nn.train(train, valid)

    gradients_analysis(nn, train_set=train)


if __name__ == "__main__":
    finite_difference_gradients()
