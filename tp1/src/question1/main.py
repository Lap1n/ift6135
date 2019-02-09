from tp1.src.question1.get_dataset import get_dataset
from tp1.src.question1.nn import NN

nn = NN()
nn.initialize_weights("normal")

dataset = get_dataset()
train = dataset[0]
valid = dataset[1]
test = dataset[2]

nn.train(train, valid)
nn.test(test)
