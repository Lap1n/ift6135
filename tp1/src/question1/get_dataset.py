import gzip
import pickle


def get_dataset():
    with gzip.open("./datasets/mnist.pkl.gz", "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
        return p
