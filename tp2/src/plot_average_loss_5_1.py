import argparse
import sys

import matplotlib.pyplot as plt
import torch

from tp2.src.models import Batch, RNN, GRU
import numpy as np
from models import make_model as TRANSFORMER

from tp2.src.utils import ptb_raw_data, ptb_iterator, repackage_hidden
import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch Penn Treebank Language Modeling')

# Arguments you may need to set to run different experiments in 4.1 & 4.2.
parser.add_argument('--rnn_result', type=str,
                    help='path to numpy array result')
parser.add_argument('--gru_result', type=str,
                    help='path to numpy array result')
parser.add_argument('--transformer_result', type=str, help='path to numpy array result')

args = parser.parse_args()


def load_result(path):
    return np.load(path)


rnn_result_array = load_result(args.rnn_result)
gru_result_array = load_result(args.gru_result)
transformer_result_array = load_result(args.transformer_result)

# Data
df = pd.DataFrame({'Step': range(rnn_result_array.shape[0]), 'RNN': rnn_result_array[:], 'GRU': gru_result_array[:],
                   'Transformer': transformer_result_array[:]})

# Data
plt.plot('Step', 'RNN', data=df)
plt.plot('Step', 'GRU', data=df)
plt.plot('Step', 'Transformer', data=df)

plt.title('Average losses over each time step')
plt.xlabel('Steps')
plt.ylabel('Average loss')
plt.legend()

plt.show()
