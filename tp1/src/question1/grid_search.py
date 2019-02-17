from itertools import product

from question1.get_dataset import get_dataset
from question1.nn import NN


def grid_search():
    hidden_dims = [(1024, 512), (512, 1024), (2048, 1024), (1024, 2048)]
    learning_rates = [0.1, 0.01, 0.001]
    batch_sizes = [100, 1000]
    results = {
        "hidden_dims": [],
        "learning_rates": [],
        "batch_sizes": [],
        "accuracy": []
    }

    dataset = get_dataset()
    train = dataset[0]
    valid = dataset[1]
    test = dataset[2]

    for hd, lr, bs in product(hidden_dims, learning_rates, batch_sizes):
        print("----------------------------")
        params_info = "Hidden dims = {}, learning rate = {}, batch size = {}".format(hd, lr, bs)
        print(params_info)
        nn = NN(hidden_dims=hd, learning_rate=lr, mini_batch_size=bs)
        nn.initialize_weights("glorot")
        nn.train(train, valid)
        _, test_accuracy = nn.test(test)
        print(params_info, "accuracy = {}".format(test_accuracy))

        results["hidden_dims"].append(hd)
        results["learning_rates"].append(lr)
        results["batch_sizes"].append(bs)
        results["accuracy"].append(test_accuracy)

    print(results)


if __name__ == "__main__":
    grid_search()
