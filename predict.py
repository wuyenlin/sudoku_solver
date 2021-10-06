#!/usr/bin/python3
from torchvision.transforms.transforms import CenterCrop
from utils.model import Net

import torch
from torchvision import transforms
from PIL import Image


def tell_digit(model, cell: Image) -> int:
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop([80,80]),
        transforms.Resize([28,28]),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    img = transform(cell)
    img = img.unsqueeze(0)
    output = model(img)
    return torch.argmax(output).item()

def main(path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net()
    model.load_state_dict(torch.load("./mnist_cnn.pth", map_location=device))
    model = model.to(device)
    model.eval()

    original_img = Image.open(path)
    print(original_img.size)
    assert original_img.size == (900,900)
    total_grid = []
    for i in range(9):
        row = []
        for j in range(9):
            left = j*100
            right = (j+1)*100
            top = i*100
            bottom = (i+1)*100
            per_cell = original_img.crop((left, top, right, bottom))
            # per_cell.show()
            number = tell_digit(model, per_cell)
            row.append(number)
        total_grid.append(row)
    
    print(total_grid)


if __name__ == "__main__":
    path = "./doc/cropped.jpg"
    main(path)