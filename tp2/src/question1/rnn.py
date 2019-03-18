# Problem 1
import math

import torch

from torch import nn


# ----------------------------------------------------------------------------------
# The encodings of elements of the input sequence

class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        # print (x)
        return self.lut(x) * math.sqrt(self.n_units)


class HiddenLayerBlock(nn.Module):
    def __init__(self, in_size, hidden_size, dp_keep_prob):
        super(HiddenLayerBlock, self).__init__()
        # TODO: should I dropout on U?
        self.u = nn.Sequential(nn.Linear(in_size, hidden_size), nn.Dropout(1 - dp_keep_prob))
        self.w = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, inputs, hidden):
        a = self.w(hidden) + self.u(inputs)
        h = self.tanh(a)
        return h


class RNN(nn.Module):  # Implement a stacked vanilla RNN with Tanh nonlinearities.
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        """
        emb_size:     The number of units in the input embeddings
        hidden_size:  The number of hidden units per layer
        seq_len:      The length of the input sequences
        vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
        num_layers:   The depth of the stack (i.e. the number of hidden layers at
                      each time-step)
        dp_keep_prob: The probability of *not* dropping out units in the
                      non-recurrent connections.
                      Do not apply dropout on recurrent connections.
        """
        super(RNN, self).__init__()

        # TODO ========================
        # Initialization of the parameters of the recurrent and fc layers.
        # Your implementation should support any number of stacked hidden layers
        # (specified by num_layers), use an input embedding layer, and include fully
        # connected layers with dropout after each recurrent layer.
        # Note: you may use pytorch's nn.Linear, nn.Dropout, and nn.Embedding
        # modules, but not recurrent modules.
        #
        # To create a variable number of parameter tensors and/or nn.Modules
        # (for the stacked hidden layer), you may need to use nn.ModuleList or the
        # provided clones function (as opposed to a regular python list), in order
        # for Pytorch to recognize these parameters as belonging to this nn.Module
        # and compute their gradients automatically. You're not obligated to use the
        # provided clones function.
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        # TODO: Not sure about the parmeters of the embeddings
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)
        # self.embedding = WordEmbedding(emb_size, vocab_size)
        self.stacked_hidden_layers = nn.ModuleList()
        self.stacked_hidden_layers.append(HiddenLayerBlock(emb_size, hidden_size, dp_keep_prob))
        for i in range(num_layers - 1):
            self.stacked_hidden_layers.append(HiddenLayerBlock(hidden_size, hidden_size, dp_keep_prob))

        self.v = nn.Sequential(nn.Linear(hidden_size, vocab_size), nn.Dropout(1 - dp_keep_prob))
        self.soft_max = nn.Softmax()

    def init_weights(self):
        # TODO ========================
        # Initialize the embedding and output weights uniformly in the range [-0.1, 0.1]
        # and output biases to 0 (in place). The embeddings should not use a bias vector.
        # Initialize all other (i.e. recurrent and linear) weights AND biases uniformly
        # in the range [-k, k] where k is the square root of 1/hidden_size

        # TODO: INITIALIZE BIAS ?

        lower_bound = -0.1
        higher_bound = 0.1
        nn.init.uniform(self.embedding.weight, lower_bound, higher_bound)
        nn.init.uniform(self.v.weight, lower_bound, higher_bound)
        self.v.bias.data.fill_(0.0)
        self.stacked_hidden_layers.apply(self.apply_hidden_layers_init)

    def apply_hidden_layers_init(self, m):
        if type(m) == nn.Linear:
            k = (1.0 / self.hidden_size)
            torch.nn.init.uniform(m.weight, -k, k)
            torch.nn.init.uniform(m.bias, -k, k)

    def init_hidden(self):
        # TODO ========================
        # initialize the hidden states to zero
        """
        This is used for the first mini-batch in an epoch, only.
        """

        initial_hidden = torch.zeros((self.num_layers, self.batch_size, self.hidden_size))
        return initial_hidden  # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)

    def forward(self, inputs, hidden):
        # TODO ========================
        # Compute the forward pass, using nested python for loops.
        # The outer for loop should iterate over timesteps, and the
        # inner for loop should iterate over hidden layers of the stack.
        #
        # Within these for loops, use the parameter tensors and/or nn.modules you
        # created in __init__ to compute the recurrent updates according to the
        # equations provided in the .tex of the assignment.
        #
        # Note that those equations are for a single hidden-layer RNN, not a stacked
        # RNN. For a stacked RNN, the hidden states of the l-th layer are used as
        # inputs to to the {l+1}-st layer (taking the place of the input sequence).

        """
        Arguments:
            - inputs: A mini-batch of input sequences, composed of integers that
                        represent the index of the current token(s) in the vocabulary.
                            shape: (seq_len, batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)

        Returns:
            - Logits for the softmax over output tokens at every time-step.
                  **Do NOT apply softmax to the outputs!**
                  Pytorch's CrossEntropyLoss function (applied in question1.py) does
                  this computation implicitly.
                        shape: (seq_len, batch_size, vocab_size)
            - The final hidden states for every layer of the stacked RNN.
                  These will be used as the initial hidden states for all the
                  mini-batches in an epoch, except for the first, where the return
                  value of self.init_hidden will beh used.
                  See the repackage_hiddens function in question1.py for more details,
                  if you are curious.
                        shape: (num_layers, batch_size, hidden_size)
        """
        # logits = torch.empty((self.seq_len, self.batch_size, self.vocab_size), requires_grad=True).cuda()
        logits = []

        for time_step in range(self.seq_len):
            new_hidden_current_step = []
            input_current_time_step = inputs[time_step]
            embedding_out = self.embedding(input_current_time_step)
            new_hidden_current_step.append(self.stacked_hidden_layers[0](embedding_out, hidden[0]))
            for i in range(1, len(self.stacked_hidden_layers)):
                new_hidden_current_step.append(self.stacked_hidden_layers[i](new_hidden_current_step[i - 1], hidden[i]))
                logits.append(self.v(new_hidden_current_step[-1]))

            # logits[time_step] = self.v(hidden[-1]).detach().requires_grad_()
        logits = torch.stack(logits)
        hidden = torch.stack(new_hidden_current_step)
        return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

    def generate(self, input, hidden, generated_seq_len):
        # TODO ========================
        # Compute the forward pass, as in the self.forward method (above).
        # You'll probably want to copy substantial portions of that code here.
        #
        # We "seed" the generation by providing the first inputs.
        # Subsequent inputs are generated by sampling from the output distribution,
        # as described in the tex (Problem 5.3)
        # Unlike for self.forward, you WILL need to apply the softmax activation
        # function here in order to compute the parameters of the categorical
        # distributions to be sampled from at each time-step.

        """
        Arguments:
            - input: A mini-batch of input tokens (NOT sequences!)
                            shape: (batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
            - generated_seq_len: The length of the sequence to generate.
                           Note that this can be different than the length used
                           for training (self.seq_len)
        Returns:
            - Sampled sequences of tokens
                        shape: (generated_seq_len, batch_size)
        """
        # TODO: NOT SURE HERE
        samples = torch.empty((generated_seq_len, self.batch_size))
        input_current_time_step = input
        for time_step in range(generated_seq_len):
            embedding_out = self.embedding(input_current_time_step)
            hidden[0] = self.stacked_hidden_layers[0](embedding_out, hidden[0])
            for i in range(1, len(self.stacked_hidden_layers)):
                hidden[i] = self.stacked_hidden_layers[i](embedding_out, hidden[i])
            logits = self.soft_max(self.v(hidden[-1]))
            input_current_time_step = torch.max(logits, 1)
            samples[time_step] = input_current_time_step

        return samples
