import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import CLEVR, collate_data, transform
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

N_VIS_ITEMS = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_metric(name, *tensors):
    print("-"*45)
    print(name)
    print("-"*45)
    print(*tensors)

def visualize(net, clevr_dir, batch_size=64):
    net.train(False)
    clevr = CLEVR(clevr_dir, "val", transform=None)
    test_set = DataLoader(clevr, batch_size=batch_size, num_workers=4, collate_fn=collate_data)
    dataset = iter(test_set)

    net.save_states = True
    # # Writer will output to ./runs/ directory by default
    # writer = SummaryWriter()

    for i in range(N_VIS_ITEMS):
        (image, question, q_len, answer, _) = next(dataset)
        image, question, answer = (image.to(device), question.to(device), answer.to(device))
        output, states = net(image, question, q_len)
        # print_metric("Output stats", output.mean(0), answer, output.shape, answer.shape)
        states["grn"] = torch.stack(states["grn"], 1)#[b, t, n, f]
        mean_state_changes = state_change_across_time(states)
        print_metric("Metric of state change over timesteps", mean_state_changes)

        # per_neuron_weights = intra_neuron_weights(states, net)
        # print_metric("Intra-neuron weights", per_neuron_weights)

def state_change_across_time(hidden_states):
    batch_states = hidden_states["grn"]
    print(torch.round(torch.mean(batch_states[0], -1)*10))
    state_change_across_time = batch_states - batch_states.mean(1).unsqueeze(1) # [b, 1, n, f]
    state_change_across_time = torch.round(state_change_across_time.mean(3).mean(0)*10)/10
    return state_change_across_time

# def intra_neuron_weights(hidden_states, net):
#     batch_states = hidden_states["grn"]
#     message_filters = net.grn.filter_gen(hidden_states["grn"], net.grn.edge_matrix)
#     per_neuron_weights = []
#     for neuron_matrix in message_filters:
#         incoming_weights = F.softmax(neuron_matrix.mean((1,2)), 0)
#         per_neuron_weights.append(incoming_weights)
#     per_neuron_weights = torch.stack(per_neuron_weights, 0)
#     return per_neuron_weights


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(
        bottom=-0.001, top=np.quantile(torch.tensor(max_grads).cpu(), 0.75)
    )  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )
    plt.show()