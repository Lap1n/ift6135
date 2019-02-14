from itertools import product

from question1.get_dataset import get_dataset
from question1.nn import NN


def main():
    nn = NN()
    nn.initialize_weights("glorot")

    dataset = get_dataset()
    train = dataset[0]
    valid = dataset[1]
    test = dataset[2]

    nn.train(train, valid)
    nn.test(test)


if __name__ == "__main__":
    main()

