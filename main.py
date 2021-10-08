#!/usr/bin/python3
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

from utils.model import Net

"""
This script is borrowed and modified from
https://github.com/pytorch/examples/blob/master/mnist/main.py
Since the original neural network was giving low accuracy, 
I have built my own model in `utils.model.py`.
"""


def train(args, model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    correct = 0
    for data, label in train_loader:
        data, label = data.to(device), label.to(device)  
        optimizer.zero_grad()  
        output = model(data) 
        loss = criterion(output, label)  
        loss.backward() 
        optimizer.step()  
        train_loss += loss.item() 
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()

    accuracy = correct*100 / len(train_loader.dataset)
    print("epoch for train: {}, accuracy: ({:.2f}%)".format(epoch, accuracy))
    return accuracy



def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct*100 / len(test_loader.dataset)
    print("epoch for test: {}, accuracy: ({:.2f}%)".format(epoch, accuracy))
    return accuracy


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--bs", type=int, default=128, metavar="N",
                        help="input batch size for training (default: 128)")
    parser.add_argument("--test_bs", type=int, default=1000, metavar="N",
                        help="input batch size for testing (default: 1000)")
    parser.add_argument("--epochs", type=int, default=20, metavar="N",
                        help="number of epochs to train (default: 20)")
    parser.add_argument("--lr", type=float, default=1e-1, metavar="LR",
                        help="learning rate (default: 1e-04)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M",
                        help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--no_cuda", action="store_true", default=False,
                        help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar="S",
                        help="random seed (default: 1)")
    parser.add_argument("--log_interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--export_training_curves", action="store_true", 
                        help="Save train/val curves in .png file")
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.bs, "shuffle": True}
    test_kwargs = {"batch_size": args.test_bs, "shuffle": True}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1,
                       "pin_memory": True,
                       "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize([28,28]),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST("../data", train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST("../data", train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_accuracy = []
    test_accuracy = []
    for epoch in range(1, args.epochs + 1):

        train_acc = train(args, model, device, train_loader, optimizer, criterion, epoch)
        test_acc = test(model, device, test_loader, epoch)
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)

        if args.export_training_curves and epoch > 3:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use("Agg")
            plt.figure()
            epoch_x = np.arange(3, len(train_accuracy)) + 1
            plt.plot(epoch_x, train_accuracy, "--", color="C0")
            plt.plot(epoch_x, test_accuracy, color="C1")
            plt.legend(["", ""])
            plt.ylabel("Accuracy (%)")
            plt.xlabel("Epoch")
            plt.xlim((3, epoch))
            plt.savefig("./accuracy.png")
            plt.close("all")

    save_path = "./mnist_cnn.pth"
    torch.save(model.state_dict(), save_path)
    print("Parameters saved to ", save_path)


if __name__ == "__main__":
    main()