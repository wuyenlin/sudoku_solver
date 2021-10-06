#!/usr/bin/python3
from utils.model import Net

import torch
from torchvision import transforms
from PIL import Image

def crop_subgrid():
    import cv2 as cv
    img = cv.imread("./doc/cropped.jpg")
    print(img.shape)
    cropped = img[400:500, 400:500, :]
    cv.imshow("image", img[400:500, 400:500, :])
    cv.imwrite("./data/6.jpg", cropped)
    k = cv.waitKey(0) & 0xFF
    if k == 27:
        cv.destroyAllWindows()


def tell_digit(model, cell: Image) -> int:
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
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
            number = tell_digit(model, per_cell)
            row.append(number)
        total_grid.append(row)
    
    print(total_grid)


if __name__ == "__main__":
    path = "./doc/cropped.jpg"
    main(path)