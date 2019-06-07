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
        self.query = linear(state_dim, image_feature_dim)
        self.concat = linear(image_feature_dim * 2, image_feature_dim)
        self.attn = linear(image_feature_dim, 1)
        self.out = linear(image_feature_dim, state_dim)

    def forward(self, image, in_state):
        # transform input from neuron into query (control+memory in MAC)
        query = self.query(in_state)
        # combine query with the image, and just the image as a bonus
        # permute to (batch, image_feature_dim, h*w)
        # this step may not be necessary
        concat = self.concat(torch.cat([query * image, image], 1)).permute(0, 2, 1)

        attn = self.attn(concat).squeeze(2)  # generate featurewise attn
        attn = F.softmax(attn, 1).unsqueeze(1)  # softmax featurewise attns

        # attn shape is (b, 1, h*w)

        # save attentions from this step for visualization
        # self.saved_attns.append(attn)

        # sum over pixels to give (b, image_feature_dim)
        out = (attn * image).sum(2)
        return self.out(out)


class TextAttnModule(nn.Module):
    def __init__(self, state_dim, text_feature_dim=512):
        super().__init__()

        self.query = linear(state_dim, text_feature_dim)
        self.query_question = linear(state_dim * 2, state_dim)
        self.attn = linear(state_dim, 1)
        self.out = linear(text_feature_dim, state_dim)

        self.state_dim = state_dim

    def forward(self, context, question, in_state):
        query = self.query(in_state)

        query_question = torch.cat([query, question], 1)
        query_question = self.query_question(query_question).unsqueeze(1)

        context_prod = query_question * context
        attn_weight = self.attn(context_prod)

        attn = F.softmax(attn_weight, 1)

        out = (attn * context).sum(1)

        return self.out(out)
