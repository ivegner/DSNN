import torch
from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_, normal
import torch.nn.functional as F

import numpy as np


class ConvFilterGenerator(nn.Module):
    """A model mapping each edge label value to a matrix or vector as the basis for message generation.
    Message generation works similar to regular convolutions for images with a discrete pixel grid.
    Here however, edge vectors are continuous which makes such a filter generator necessary.
    Depending on the method (the continuous convolutions in SchNet or EdgeNet in the MPNN paper),
    either a vector or a matrix are output for each edge.
    """

    def __init__(
        self,
        n_neurons,
        state_dim,
        edge_dim,
        message_dim=32,  # only used if use_matrix_filters==True
        n_filter_nn_layers=2,
        use_matrix_filters=True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.use_matrix_filters = use_matrix_filters
        self.n_neurons = n_neurons
        self.message_dim = message_dim
        self.edge_dim = edge_dim
        self.n_hidden_filter_layers = n_filter_nn_layers

        out_flat_dim = state_dim * message_dim if use_matrix_filters else state_dim
        # "smooth" interpolation of neurons
        hidden_layer_units = np.linspace(
            start=state_dim + edge_dim, stop=out_flat_dim, num=n_filter_nn_layers + 2, dtype=int
        ).tolist()

        # assemble layers
        layers = []
        for i, layer_units in enumerate(hidden_layer_units):
            if i < len(hidden_layer_units) - 1:
                layers.append(linear(layer_units, hidden_layer_units[i + 1]))
                layers.append(nn.SELU())

        self.filter_gen_fc = nn.Sequential(*layers)

    def forward(self, hidden_states, edge_matrix):
        """Forward pass of filter generator.
        :param edge_matrix: edge matrix shaped [n_neurons, n_neurons, edge_dim]
        :return: generated message transformations. If self.use_matrix_filters is True, the shape is
            [n_neurons, n_neurons, state_dim, message_dim],
            else we have a vector for each neuron pair: [n_neurons, n_neurons, state_dim]
        """
        # NO BATCHING BECAUSE WE USE THE SAME MATRICES FOR EVERY SAMPLE IN THE ENTIRE BATCH
        batch_size = hidden_states.size(0)
        concat = torch.cat(
            [
                edge_matrix.expand((batch_size, -1, -1, -1)),
                hidden_states.unsqueeze(2).expand((-1, -1, self.n_neurons, -1)),
            ],
            dim=-1,
        )
        concat_flat = concat.view(-1, (self.edge_dim + self.state_dim))
        message_matrices = self.filter_gen_fc(concat_flat)
        if self.use_matrix_filters:
            message_matrices = message_matrices.view(
                [batch_size, self.n_neurons, self.n_neurons, self.message_dim, self.state_dim]
            )
        else:
            message_matrices = message_matrices.view(
                [batch_size, self.n_neurons, self.n_neurons, self.state_dim]
            )

        return message_matrices


class MatrixMessagePassing(nn.Module):
    """Implements EdgeNetwork message function from MPNN paper.
    To generate the message from neuron j to neuron i, the message matrix belonging to the edge vector between i and j
    is multiplied with the hidden state of j. All messages to neuron i are summed and a bias is added.
    """

    def __init__(self, state_dim: int, message_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.message_dim = message_dim

        self.message_bias = nn.Parameter(torch.zeros(message_dim))

    def forward(self, hidden_states: torch.Tensor, matrix_filters: torch.Tensor):
        """Forward pass for generating messages using matrix filters.
        :param hidden_states: Hidden states of all neurons, shaped [batch_size, n_neurons, state_dim]
        :param matrix_filters: generated convolution filters, shaped
            [batch_size, n_neurons, n_neurons, state_dim, state_dim]
        :return: sum of incoming messages to each neuron, shaped [batch_size, n_neurons, state_dim]
        """
        batch_size = hidden_states.shape[0]
        n_neurons = hidden_states.shape[1]

        # multiply matrix with hidden states and add bias to generate messages
        hidden_states_flat = hidden_states.view([batch_size, n_neurons * self.state_dim, 1])

        matrix_filters = matrix_filters.permute([0, 1, 3, 2, 4])
        matrix_filters = matrix_filters.contiguous().view(
            [batch_size, n_neurons * self.message_dim, n_neurons * self.state_dim]
        )

        # (1, n*m_d, n*d) x (b, n*d, 1) -> (b, n*m_d, 1)
        messages = matrix_filters @ hidden_states_flat
        messages = messages.view([batch_size, n_neurons, self.message_dim])
        messages += self.message_bias

        return messages


class VectorMessagePassing(nn.Module):
    """Messages are generated by elementwise product with a vector instead of a product with a matrix, as in SchNet.
    To generate the message from neuron j to neuron i, the filter vector belonging to the edge between i and j
    is multiplied elementwise with the hidden state of j. The total message to neuron i is the sum of all incoming ones.
    """

    def forward(self, hidden_states, vector_transforms):
        """Forward pass for generating messages using vector filters.
        :param hidden_states: Hidden states of all neurons, shaped [batch_size, n_neurons, state_dim]
        :param vector_transforms: generated message transforms shaped
            [batch_size, n_neurons, n_neurons, state_dim]
        :return: sum of incoming messages to each neuron, shaped [batch_size, n_neurons, state_dim]
        """
        hidden_states = torch.unsqueeze(hidden_states, 1)
        messages = hidden_states * vector_transforms
        messages = torch.sum(
            messages, 2
        )  # total message to neuron i = sum over messages from neurons j

        return messages


def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    kaiming_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()

    return lin


class GRUUpdate(nn.Module):
    """Implements the GRU Update function to update the hidden states based on their incoming messages.
    """

    def __init__(self, state_dim, message_dim):
        super().__init__()
        self.gru = nn.GRU(message_dim, state_dim)
        self.message_dim = message_dim
        self.state_dim = state_dim

    def forward(self, hidden_states, messages):
        """Forward pass updating each hidden state using its incoming messages.
        In contrast to the original definition, we only use one message per graph edge.
        :param hidden_states: Hidden states of all neurons, shaped [batch_size, n_neurons, state_dim]
        :param messages: sum of incoming messages for each neuron, shaped [batch_size, n_neurons, state_dim]
        :param mask: indicates whether a neuron is actually present (1) or zero-padded (0). [batch_size, n_neurons]
        :return: updated states shaped [batch_size, n_neurons, state_dim]
        """
        batch_size = hidden_states.shape[0]
        n_neurons = hidden_states.shape[1]

        # reshape hidden states, messages and mask so that one batch = one neuron
        hidden_states = hidden_states.view([1, batch_size * n_neurons, self.state_dim])
        messages = messages.view([1, batch_size * n_neurons, self.message_dim])
        # mask = tf.cast(tf.reshape(mask, [batch_size * n_neurons, 1]), tf.float32)

        updated_states = self.gru(messages, hidden_states)[1]
        # # zero out masked nodes
        # updated_states = updated_states * mask

        # reshape back to original shape
        updated_states = updated_states.view([batch_size, n_neurons, self.state_dim])

        return updated_states
