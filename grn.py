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
    def __init__(
        self,
        n_neurons,
        state_dim,
        edge_dim=5,
        message_dim=32,
        max_step=12,
        matrix_messages=True,
        batch_size=64,
    ):
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

        self.hidden_states = torch.zeros((batch_size, n_neurons, n_neurons, state_dim))
        self.max_step = max_step
        self.n_neurons = n_neurons
        self.state_dim = state_dim
        self.batch_size = batch_size

    def forward(self, image, question):
        """Forward pass of the message passing neural network."""

        filters = self.filter_gen.forward(self.edge_matrix)
        # perform message passing
        for i in range(self.max_step):
            messages = self.message_passing.forward(self.hidden_states, filters)
            hidden_states = self.update.forward(hidden_states, messages)

        return output


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
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1024, image_feature_dim, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(image_feature_dim, image_feature_dim, 3, padding=1),
            nn.ELU(),
        )

        self.embed = nn.Embedding(n_vocab, embed_hidden)
        self.lstm = nn.LSTM(embed_hidden, text_feature_dim, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(text_feature_dim * 2, text_feature_dim)

        self.submodules = nn.ModuleDict(
            [
                ("image_attn", ImageAttnModule(state_dim, image_feature_dim=image_feature_dim)),
                ("text_attn", TextAttnModule(state_dim, text_feature_dim=text_feature_dim)),
            ]
        )

        self.grn = GRN(
            n_neurons,
            state_dim,
            edge_dim=5,
            message_dim=32,
            max_step=12,
            matrix_messages=True,
            batch_size=batch_size,
        )

        self.classifier = nn.Sequential(
            linear(state_dim, state_dim), nn.ELU(), linear(state_dim, classes)
        )

        self.max_step = max_step
        self.state_dim = state_dim
        self.batch_size = batch_size

        self.image_feature_dim = image_feature_dim
        self.text_feature_dim = text_feature_dim

        self.reset()

    def reset(self):
        self.embed.weight.data.uniform_(0, 1)

        kaiming_uniform_(self.conv[0].weight)
        self.conv[0].bias.data.zero_()
        kaiming_uniform_(self.conv[2].weight)
        self.conv[2].bias.data.zero_()

        kaiming_uniform_(self.classifier[0].weight)

    def forward(self, image, question, question_len, dropout=0.15):
        batch_size = question.size(0)

        img = self.conv(image)
        img = img.view(batch_size, self.image_feature_dim, -1)

        embed = self.embed(question)
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len, batch_first=True)
        lstm_out, (h, _) = self.lstm(embed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.lstm_proj(lstm_out)
        h = h.permute(1, 0, 2).contiguous().view(batch_size, -1)

        # Run GRN classifier
        output = self.grn(lstm_out, h, img)
        output = torch.cat(output, 1)

        # Read out output
        out = torch.cat([output, h], 1)
        out = self.classifier(out)

        return out
