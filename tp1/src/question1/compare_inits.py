from question1.get_dataset import get_dataset
from question1.nn import NN


def compare_inits():
    inits = ["zero", "normal", "glorot"]

    dataset = get_dataset()
    train = dataset[0]
    valid = dataset[1]
    test = dataset[2]

    for init in inits:
        print("Training with {} initialisation".format(init))
        nn = NN()
        nn.initialize_weights(init)

        nn.train(train, valid)
        nn.test(test)
        nn.save_results("results/{}/".format(init))


if __name__ == "__main__":
    compare_inits()
