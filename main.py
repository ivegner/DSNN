import sys
import os
import pickle
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import click
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from dataset import CLEVR, collate_data, transform
from grn import GRNModel

batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def train(net, optimizer, criterion, clevr_dir, epoch):
    clevr = CLEVR(clevr_dir, transform=transform)
    train_set = DataLoader(clevr, batch_size=batch_size, num_workers=4, collate_fn=collate_data)

    dataset = iter(train_set)
    pbar = tqdm(dataset)
    moving_loss = 0

    net.train(True)
    for i, (image, question, q_len, answer, _) in enumerate(pbar):
        image, question, answer = (image.to(device), question.to(device), answer.to(device))

        net.zero_grad()
        output = net(image, question, q_len)
        loss = criterion(output, answer)
        loss.backward()

        # if wrapped in a DataParallel, the actual net is at DataParallel.module
        # m = net.module if isinstance(net, nn.DataParallel) else net
        # torch.nn.utils.clip_grad_norm_(m.parameters(), 1)
        # torch.nn.utils.clip_grad_value_(net.parameters(), 0.05)
        # print("GRADS", dict(net.named_parameters())["module.grn.filter_gen.filter_gen_fc.0.weight"].grad)
        # if i % 100 == 0:
        #     plot_grad_flow(net.named_parameters())

        optimizer.step()
        correct = output.detach().argmax(1) == answer
        correct = torch.tensor(correct, dtype=torch.float32).sum() / batch_size

        if moving_loss == 0:
            moving_loss = correct

        else:
            moving_loss = moving_loss * 0.99 + correct * 0.01

        pbar.set_description(
            "Epoch: {}; Loss: {:.5f}; Acc: {:.5f}".format(epoch + 1, loss.item(), moving_loss)
        )

    clevr.close()


def valid(net, clevr_dir, epoch):
    clevr = CLEVR(clevr_dir, "val", transform=None)
    valid_set = DataLoader(clevr, batch_size=batch_size, num_workers=4, collate_fn=collate_data)
    dataset = iter(valid_set)

    net.train(False)
    family_correct = Counter()
    family_total = Counter()
    with torch.no_grad():
        for image, question, q_len, answer, family in tqdm(dataset):
            image, question = image.to(device), question.to(device)

            output = net(image, question, q_len)
            correct = output.detach().argmax(1) == answer.to(device)
            for c, fam in zip(correct, family):
                if c:
                    family_correct[fam] += 1
                family_total[fam] += 1

    with open("log/log_{}.txt".format(str(epoch + 1).zfill(2)), "w") as w:
        for k, v in family_total.items():
            w.write("{}: {:.5f}\n".format(k, family_correct[k] / v))

    print("Avg Acc: {:.5f}".format(sum(family_correct.values()) / sum(family_total.values())))

    clevr.close()


def test(net, clevr_dir):
    print("Starting tests!")
    print(net)
    clevr = CLEVR(clevr_dir, "val", transform=None)
    test_set = DataLoader(clevr, batch_size=batch_size, num_workers=4, collate_fn=collate_data)
    dataset = iter(test_set)

    net.train(False)
    family_correct = Counter()
    family_total = Counter()
    with torch.no_grad():
        for image, question, q_len, answer, family in tqdm(dataset):
            image, question = image.to(device), question.to(device)

            output = net(image, question, q_len)

            # if wrapped in a DataParallel, the actual net is at DataParallel.module
            m = net.module if isinstance(net, nn.DataParallel) else net
            # [{read, write}, n_steps, batch_size, {??????, n_memories}]

            # TESTS HERE

            correct = output.detach().argmax(1) == answer.to(device)
            for c, fam in zip(correct, family):
                if c:
                    family_correct[fam] += 1
                family_total[fam] += 1

    with open("log/test_log.txt", "w") as w:
        for k, v in family_total.items():
            w.write("{}: {:.5f}\n".format(k, family_correct[k] / v))

    print("Avg Acc: {:.5f}".format(sum(family_correct.values()) / sum(family_total.values())))

    clevr.close()


@click.command()
@click.argument("clevr_dir")
@click.option(
    "-n", "--n-neurons", default=3, show_default=True, help="Number of neurons for the network"
)
@click.option(
    "-h", "--hidden-dim", default=512, show_default=True, help="Hidden dimensions for the neurons"
)
@click.option("-l", "--load", "load_filename", type=str, help="load a model")
@click.option("-e", "--n-epochs", default=20, show_default=True, help="Number of epochs")
@click.option(
    "-t",
    "--only-test",
    default=False,
    is_flag=True,
    show_default=True,
    help="Do not train. Only run tests and export results for visualization.",
)
def main(clevr_dir, n_neurons, hidden_dim=512, load_filename=None, n_epochs=20, only_test=False):
    with open(os.path.join(clevr_dir, "preprocessed/dic.pkl"), "rb") as f:
        dic = pickle.load(f)

    n_words = len(dic["word_dic"]) + 1
    n_answers = len(dic["answer_dic"])

    net = GRNModel(
        n_words,
        n_neurons,
        hidden_dim,
        image_feature_dim=hidden_dim,
        text_feature_dim=hidden_dim,
        message_dim=hidden_dim,
        edge_dim=5,
    )
    net = net.to(device)
    print(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    start_epoch = 0

    if device.type == "cuda":
        devices = [0] if only_test else None
        print("Using", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net, device_ids=devices)

    if load_filename:
        checkpoint = torch.load(load_filename)
        # new format
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Starting at epoch {start_epoch}")

    if not only_test:
        # do training and validation
        for epoch in range(start_epoch, n_epochs):
            train(net, optimizer, criterion, clevr_dir, epoch)
            valid(net, clevr_dir, epoch)

            with open(
                f"checkpoint/checkpoint_{str(epoch + 1).zfill(2)}_{n_neurons}m_{hidden_dim}h.model", "wb"
            ) as f:

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": (
                            net.module if isinstance(net, nn.DataParallel) else net
                        ).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    f,
                )
    else:
        # predict on the test set and make visualization data
        test(net, clevr_dir, batch_size=batch_size)


if __name__ == "__main__":
    main()
