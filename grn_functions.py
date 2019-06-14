import torch
from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_, normal
import torch.nn.functional as F

import numpy as np


# class ConvFilterGenerator(nn.Module):
#     """A model mapping each edge label value to a matrix or vector as the basis for message generation.
#     Message generation works similar to regular convolutions for images with a discrete pixel grid.
#     Here however, edge vectors are continuous which makes such a filter generator necessary.
#     Depending on the method (the continuous convolutions in SchNet or EdgeNet in the MPNN paper),
#     either a vector or a matrix are output for each edge.
#     """

#     def __init__(
#         self,
#         n_neurons,
#         state_dim,
#         edge_dim,
#         message_dim=32,  # only used if use_adjacency_matrix==True
#         n_filter_nn_layers=2,
#         use_adjacency_matrix=True,
#     ):
#         super().__init__()
#         self.state_dim = state_dim
#         self.use_adjacency_matrix = use_adjacency_matrix
#         self.n_neurons = n_neurons
#         self.message_dim = message_dim
#         self.edge_dim = edge_dim
#         self.n_hidden_filter_layers = n_filter_nn_layers

#         out_flat_dim = state_dim * message_dim if use_adjacency_matrix else state_dim
#         # "smooth" interpolation of neurons
#         hidden_layer_units = np.linspace(
#             start=state_dim + edge_dim, stop=out_flat_dim, num=n_filter_nn_layers + 2, dtype=int
#         ).tolist()

#         # assemble layers
#         layers = []
#         for i, layer_units in enumerate(hidden_layer_units):
#             if i < len(hidden_layer_units) - 1:
#                 layers.append(linear(layer_units, hidden_layer_units[i + 1]))
#                 layers.append(nn.SELU())

#         self.filter_gen_fc = nn.Sequential(*layers)

#     def forward(self, hidden_states, edge_matrix):
#         """Forward pass of filter generator.
#         :param edge_matrix: edge matrix shaped [n_neurons, n_neurons, edge_dim]
#         :return: generated message transformations. If self.use_adjacency_matrix is True, the shape is
#             [n_neurons, n_neurons, state_dim, message_dim],
#             else we have a vector for each neuron pair: [n_neurons, n_neurons, state_dim]
#         """
#         # NO BATCHING BECAUSE WE USE THE SAME MATRICES FOR EVERY SAMPLE IN THE ENTIRE BATCH
#         batch_size = hidden_states.size(0)
#         concat = torch.cat(
#             [
#                 edge_matrix.expand((batch_size, -1, -1, -1)),
#                 hidden_states.unsqueeze(2).expand((-1, -1, self.n_neurons, -1)),
#             ],
#             dim=-1,
#         )
#         concat_flat = concat.view(-1, (self.edge_dim + self.state_dim))
#         message_matrices = self.filter_gen_fc(concat_flat)
#         if self.use_adjacency_matrix:
#             message_matrices = message_matrices.view(
#                 [batch_size, self.n_neurons, self.n_neurons, self.message_dim, self.state_dim]
#             )
#         else:
#             message_matrices = message_matrices.view(
#                 [batch_size, self.n_neurons, self.n_neurons, self.state_dim]
#             )

#         return message_matrices


class MatrixMessagePassing(nn.Module):
    """Implements EdgeNetwork message function from MPNN paper.
    To generate the message from neuron j to neuron i, the message matrix belonging to the edge vector between i and j
    is multiplied with the hidden state of j. All messages to neuron i are summed and a bias is added.
    """

    def __init__(self, state_dim: int, n_neurons:int):
        super().__init__()
        self.state_dim = state_dim

        self.message_bias = nn.Parameter(torch.zeros(state_dim))
        self.eye = nn.Parameter(torch.eye(n_neurons), requires_grad=False)

    def forward(self, hidden_states: torch.Tensor, adjacency_matrix: torch.Tensor):
        """Forward pass for generating messages using matrix filters.
        :param hidden_states: Hidden states of all neurons, shaped [batch_size, n_neurons, state_dim]
        :param adjacency_matrix: adjacency matrix, shaped
            [n_neurons, n_neurons]
        :return: sum of incoming messages to each neuron, shaped [batch_size, n_neurons, state_dim]
        """
        batch_size = hidden_states.shape[0]
        n_neurons = hidden_states.shape[1]

        # (1, n, n) x (b, n, d) -> (b, n, d)
        messages = (self.eye - adjacency_matrix) @ hidden_states
        messages += self.message_bias

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

    def __init__(self, state_dim):
        super().__init__()
        self.gru = nn.GRU(state_dim, state_dim)
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
        messages = messages.view([1, batch_size * n_neurons, self.state_dim])
        # mask = tf.cast(tf.reshape(mask, [batch_size * n_neurons, 1]), tf.float32)

        updated_states = self.gru(messages, hidden_states)[1]
        # # zero out masked nodes
        # updated_states = updated_states * mask

        # reshape back to original shape
        updated_states = updated_states.view([batch_size, n_neurons, self.state_dim])

        return updated_states
