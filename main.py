DEBUG = False
if DEBUG:
    import multiprocessing

    multiprocessing.set_start_method("spawn", True)

import sys
import os
import pickle
from collections import Counter

import numpy as np
import torch

torch.manual_seed(2)
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import click

from dataset import CLEVR, collate_data, transform
from grn import GRNModel
from visualize import visualize, plot_grad_flow

batch_size = 64

device = (
    torch.device("cpu") if DEBUG else torch.device("cuda" if torch.cuda.is_available() else "cpu")
)


def train(net, optimizer, criterion, clevr_dir, epoch):
    clevr = CLEVR(clevr_dir, transform=transform)
    train_set = DataLoader(
        clevr,
        batch_size=batch_size,
        num_workers=0 if DEBUG else 4,
        collate_fn=collate_data,
        shuffle=True,
    )

    dataset = iter(train_set)
    pbar = tqdm(dataset)
    moving_loss = 0

    net.train(True)
    for name, param in net.named_parameters():
        if "grn" not in name and epoch < 1:
            param.requires_grad = False
        elif epoch < 3 and "classifier" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    for i, (image, question, q_len, answer, _) in enumerate(pbar):
        image, question, answer = (image.to(device), question.to(device), answer.to(device))

        net.zero_grad()
        output = net(image, question, q_len)
        loss = criterion(output, answer)
        loss.backward()

        # if wrapped in a DataParallel, the actual net is at DataParallel.module
        m = net.module if isinstance(net, nn.DataParallel) else net
        # torch.nn.utils.clip_grad_norm_(m.parameters(), 1)
        torch.nn.utils.clip_grad_value_(net.parameters(), 5)
        # print("GRADS", dict(net.named_parameters())["module.grn.filter_gen.filter_gen_fc.0.weight"].grad)
        if i % 1000 == 0:
            with torch.no_grad():
                visualize(net.module, clevr_dir, batch_size)
                plot_grad_flow(net.named_parameters())
                net.train(True)
                net.module.save_states = False

        optimizer.step()
        correct = output.detach().argmax(1) == answer
        correct = torch.tensor(correct, dtype=torch.float32).sum() / batch_size

        if moving_loss == 0:
            moving_loss = correct

        else:
            moving_loss = moving_loss * 0.99 + correct * 0.01

        pbar.set_description(
            "Epoch: {}; Loss: {:.5f}; Acc: {:.5f}".format(epoch, loss.item(), moving_loss)
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
    help="Do not train. Only run tests.",
)
@click.option(
    "--vis",
    "--only-visualize",
    "only_visualize",
    default=False,
    is_flag=True,
    show_default=True,
    help="Do not train. Only do visualization.",
)
def main(
    clevr_dir,
    n_neurons,
    hidden_dim=512,
    message_dim=32,
    load_filename=None,
    n_epochs=20,
    only_test=False,
    only_visualize=False,
):
    do_train = not (only_test or only_visualize)
    with open(os.path.join(clevr_dir, "preprocessed/dic.pkl"), "rb") as f:
        dic = pickle.load(f)

    n_words = len(dic["word_dic"]) + 1
    n_answers = len(dic["answer_dic"])

    net = GRNModel(
        n_words,
        n_neurons,
        hidden_dim,
        image_feature_dim=512,
        text_feature_dim=512,
        message_dim=message_dim,
        edge_dim=5,
        # matrix_messages=False
    )
    net = net.to(device)
    print(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    start_epoch = 0

    if load_filename:
        checkpoint = torch.load(load_filename)
        # new format
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Starting at epoch {start_epoch}")

    if device.type == "cuda" and not only_visualize:
        print("Using", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    if do_train:
        # do training and validation
        for epoch in range(start_epoch, n_epochs):
            train(net, optimizer, criterion, clevr_dir, epoch)
            valid(net, clevr_dir, epoch)

            with open(
                f"checkpoint/checkpoint_{str(epoch).zfill(2)}_{n_neurons}n_{hidden_dim}h.model",
                "wb",
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
        if only_test:
            test(net, clevr_dir, batch_size=batch_size)
        if only_visualize:
            visualize(net, clevr_dir, batch_size=batch_size)


if __name__ == "__main__":
    main()
