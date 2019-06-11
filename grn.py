import torch
from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_, normal
import torch.nn.functional as F

from grn_functions import (
    ConvFilterGenerator,
    MatrixMessagePassing,
    VectorMessagePassing,
    GRUUpdate,
    linear,
)

from modules import TextAttnModule, ImageAttnModule


class GRN(nn.Module):
    def __init__(self, n_neurons, state_dim, edge_dim=32, message_dim=32, matrix_messages=True):
        super().__init__()
        self.filter_gen = ConvFilterGenerator(
            n_neurons, state_dim, edge_dim, message_dim=message_dim, use_matrix_filters=True
        )

        if matrix_messages:
            self.message_passing = MatrixMessagePassing(state_dim, message_dim)
        else:
            self.message_passing = VectorMessagePassing()
        self.update = GRUUpdate(state_dim, message_dim)

        # learned edges
        self.edge_matrix = nn.Parameter(
            xavier_uniform_(torch.empty((n_neurons, n_neurons, edge_dim)))
        )

        self.initial_state = nn.Parameter(torch.zeros((n_neurons, state_dim)))
        self.n_neurons = n_neurons
        self.state_dim = state_dim

        # set in init for each training example
        self.n_outputs = None
        self.hidden_states = None
        self.filters = None

    def init(self, batch_size, n_outputs):
        # set initial hidden states
        self.hidden_states = self.initial_state.repeat(batch_size, 1, 1)
        # make edge filters
        self.filters = self.filter_gen.forward(self.edge_matrix)
        self.n_outputs = n_outputs

        # just the start values as outputs
        return self._get_outputs()

    def _get_outputs(self):
        # size of outputs: [batch_size, n_inputs, state_dim]
        # output[:, i] = self.hidden_states[:, n_neurons -i],
        # where i = index of corresponding input source
        out_indices = self.n_neurons - 1 - torch.tensor(range(self.n_outputs))
        outputs = self.hidden_states[:, out_indices, :].permute(1, 0, 2)
        return outputs

    def forward(self, inputs):
        """Forward pass of the message passing neural network.

            inputs: a `[batch_size, n_modules, state_dim]` tensor containing inputs to the network.
                They will be put directly in the hidden state, with the same neuron receiving
                the vector of the same index from inputs. So, neuron 0 will always receive
                `inputs[:, 0, :]`

        Returns: (outputs, hidden_states)

            outputs: a `[batch_size, n_modules, state_dim]` tensor containing outputs for the
            number of inputs passed in. So, `outputs[:,0,:]` will be the value of the neuron
            corresponding to `inputs[:, 0, :]`. In fact, `outputs[:, i, :]` will come from
            neuron `n_neurons - i`.

            hidden_states: `[batch_size, n_neurons, state_dim]` the hidden states of the neurons
        """
        n_inputs = inputs.size(1)
        input_indices = torch.tensor(range(n_inputs))

        # put the inputs in their respective neurons
        self.hidden_states[:, input_indices, :] = inputs[:, input_indices, :]

        # perform message passing
        messages = self.message_passing.forward(self.hidden_states, self.filters)
        self.hidden_states = self.update.forward(self.hidden_states, messages)

        # size of outputs: [n_outputs, batch_size, state_dim]
        return self._get_outputs(), self.hidden_states


class GRNModel(nn.Module):
    def __init__(
        self,
        n_vocab,
        n_neurons,
        state_dim,
        batch_size=64,
        embed_hidden=300,
        max_step=12,
        classes=28,
        image_feature_dim=512,
        text_feature_dim=512,
        message_dim=128,
        edge_dim=5
    ):
        super().__init__()

        self.submodules = nn.ModuleDict(
            [
                ("image_attn", ImageAttnModule(state_dim, image_feature_dim=image_feature_dim)),
                (
                    "text_attn",
                    TextAttnModule(
                        state_dim,
                        n_vocab,
                        embed_hidden=embed_hidden,
                        text_feature_dim=text_feature_dim,
                    ),
                ),
            ]
        )

        self.grn = GRN(n_neurons, state_dim, edge_dim=5, message_dim=message_dim, matrix_messages=True)

        self.classifier = nn.Sequential(
            linear(state_dim + text_feature_dim*2, state_dim), nn.ELU(), linear(state_dim, classes)
        )
        kaiming_uniform_(self.classifier[0].weight)

        self.max_step = max_step
        self.state_dim = state_dim
        self.batch_size = batch_size

        self.image_feature_dim = image_feature_dim
        self.text_feature_dim = text_feature_dim

    def forward(self, image, question, question_len, dropout=0.15):
        batch_size = question.size(0)
        self.submodules["image_attn"].set_input(image)
        self.submodules["text_attn"].set_input((question, question_len))

        n_grn_outputs = len(self.submodules) + 1  # 1 for final output
        # size = [n_outputs, batch_size, dim]
        grn_outputs = self.grn.init(batch_size, n_grn_outputs)
        grn_outputs, final_out = grn_outputs[:-1], grn_outputs[-1]
        for step in range(self.max_step):
            in_out = zip(self.submodules.values(), grn_outputs)
            # outputs for each submodule = inputs to grn
            submodule_outputs = torch.stack([m(x) for (m, x) in in_out], 0).permute(1,0,2)
            # Run GRN
            _grn_out, _ = self.grn(submodule_outputs)
            grn_outputs, final_out = _grn_out[:-1], _grn_out[-1]

        # Read out output
        hidden_state = self.submodules["text_attn"].input[1]
        out = torch.cat([final_out, hidden_state], 1)
        out = self.classifier(out)

        return out
