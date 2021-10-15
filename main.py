#!/usr/bin/python3
from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from PIL import Image

from utils.model import Net

"""
This script is borrowed and modified from
https://github.com/pytorch/examples/blob/master/mnist/main.py
Since the original neural network was giving low accuracy, 
I have built my own model in `utils/model.py`.
"""

class Data:
    def __init__(self, img_folder, train: bool, transform=None):
        self.transform = transform
        self.length = 900 if train else 100

        self.img_list = [os.path.join(img_folder, "{}.jpg".format(i%10)) for i in range(self.length)]
        self.label_list = [i%10 for i in range(self.length)]


    def __getitem__(self, index):
        img_path = self.img_list[index]
        image = self.transform(Image.open(img_path))
        label = self.label_list[index]
        return image, label

    def __len__(self):
        return self.length


def train(args, model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    correct = 0
    for image, label in train_loader:
        image, label = image.to(device), label.to(device)  
        optimizer.zero_grad()  
        output = model(image) 
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
        for image, target in test_loader:
            image, target = image.to(device), target.to(device)
            output = model(image)
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
    parser.add_argument("--bs", type=int, default=16, metavar="N",
                        help="input batch size for training (default: 128)")
    parser.add_argument("--test_bs", type=int, default=1000, metavar="N",
                        help="input batch size for testing (default: 1000)")
    parser.add_argument("--epochs", type=int, default=10, metavar="N",
                        help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=1e-03, metavar="LR",
                        help="learning rate (default: 1e-04)")
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

    train_kwargs = {"batch_size": args.bs, "shuffle": False}
    test_kwargs = {"batch_size": args.test_bs, "shuffle": True}
    if use_cuda:
        cuda_kwargs = {"num_workers": 4,
                       "pin_memory": True,
                       "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform_train = transforms.Compose([
        transforms.RandomAutocontrast(),
        transforms.RandomCrop(size=(80,80)),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize([28,28]),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])
    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize([28,28]),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])

    img_folder = "./data/"
    train_dataset = Data(img_folder, train=True, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=False)
    test_dataset = Data(img_folder, train=False, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=False)

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
            plt.legend(["Training", "Testing"])
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