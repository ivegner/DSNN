import torch
from torch import nn
import torch.nn.functional as F
from grn_functions import linear


class GRNModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None


class ImageAttnModule(nn.Module):
    def __init__(self, state_dim, image_feature_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1024, image_feature_dim, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(image_feature_dim, image_feature_dim, 3, padding=1),
            nn.ELU(),
        )
        nn.init.kaiming_uniform_(self.conv[0].weight)
        self.conv[0].bias.data.zero_()
        nn.init.kaiming_uniform_(self.conv[2].weight)
        self.conv[2].bias.data.zero_()
        self.query = linear(state_dim, image_feature_dim)
        self.concat = linear(image_feature_dim * 2, image_feature_dim)
        self.attn = linear(image_feature_dim, 1)
        self.out = linear(image_feature_dim, state_dim)

        self.image_feature_dim = image_feature_dim

        # set dynamically
        self.input = None

    def set_input(self, image):
        batch_size = image.size(0)

        image = self.conv(image)
        image = image.view(batch_size, self.image_feature_dim, -1)
        self.input = image

    def forward(self, in_state):
        image = self.input
        # transform input from neuron into query (control+memory in MAC)
        query = self.query(in_state).unsqueeze(2)
        # combine query with the image, and just the image as a bonus
        # permute to (batch, h*w, image_feature_dim)
        # this step may not be necessary
        concat = self.concat(torch.cat([query * image, image], 1).permute(0, 2, 1))

        attn = self.attn(concat).squeeze(2)  # generate featurewise attn
        attn = F.softmax(attn, 1).unsqueeze(1)  # softmax featurewise attns

        # attn shape is (b, 1, h*w)

        # save attentions from this step for visualization
        # self.saved_attns.append(attn)

        # sum over pixels to give (b, image_feature_dim)
        out = (attn * image).sum(2)
        return self.out(out)


class TextAttnModule(nn.Module):
    def __init__(self, state_dim, n_vocab, embed_hidden=300, text_feature_dim=512):
        super().__init__()

        self.query = linear(state_dim, text_feature_dim)
        self.query_question = linear(text_feature_dim + text_feature_dim * 2, text_feature_dim)
        self.attn = linear(text_feature_dim, 1)
        self.out = linear(text_feature_dim, state_dim)

        self.embed = nn.Embedding(n_vocab, embed_hidden)
        self.embed.weight.data.uniform_(0, 1)
        self.lstm = nn.LSTM(embed_hidden, text_feature_dim, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(text_feature_dim * 2, text_feature_dim)

        self.state_dim = state_dim

        # lstm_out, hidden_state
        self.input = (None, None)

    def set_input(self, input):
        question, question_len = input
        batch_size = question.size(0)
        embed = self.embed(question)
        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len, batch_first=True)
        lstm_out, (hidden_state, _) = self.lstm(embed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.lstm_proj(lstm_out)
        hidden_state = hidden_state.permute(1, 0, 2).contiguous().view(batch_size, -1)
        self.input = (lstm_out, hidden_state)

    def forward(self, in_state):
        context, question = self.input
        query = self.query(in_state)

        query_question = torch.cat([query, question], 1)
        query_question = self.query_question(query_question).unsqueeze(1)

        context_prod = query_question * context
        attn_weight = self.attn(context_prod)

        attn = F.softmax(attn_weight, 1)

        out = (attn * context).sum(1)

        return self.out(out)
