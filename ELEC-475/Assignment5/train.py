import os
import argparse
import torch
import time
import model
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms
from torchvision.transforms import functional
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from PIL import Image


def display_image(img_name, predicted_nose):
    print("displaying image")

    img = Image.open(img_name)
    img_size = img.size

    train_noses = 'data/train_noses.txt'
    image = os.path.basename(img_name)

    with open(train_noses, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.split(',')[0] == image:
                actual_nose = eval(line.split('"')[1])

    # Unnormalize the noses
    predicted_x, predicted_y = predicted_nose
    predicted_x = ((predicted_x + 1) / 2) * 256
    predicted_y = ((predicted_y + 1) / 2) * 256

    predicted_x = predicted_x * (img_size[0] / 256)
    predicted_y = predicted_y * (img_size[1] / 256)

    actual_x, actual_y = actual_nose

    print(f"predicted: ({int(predicted_x)}, {int(predicted_y)})")
    print(f"actual: {actual_nose}")

    plt.imshow(img)
    plt.scatter([actual_x], [actual_y], c='red', marker='X', label='Actual Coordinates')
    plt.scatter([predicted_x], [predicted_y], c='green', marker='X', label='Predicted Coordinates')
    plt.legend(loc='upper right', fontsize=8)
    plt.title('Image with Coordinates')

    plt.savefig(f'outputs/training/images/predicted_{image}.png')
    plt.show()
    plt.close()


class nose_dataset(Dataset):
    def __init__(self, dir_txt, dir_imgs, transform=None):
        self.dir_txt = dir_txt
        self.dir_imgs = dir_imgs
        self.points = pd.read_csv(self.dir_txt)
        self.transform = transform

    def __len__(self):
        return len(self.points)

    def __getitem__(self, item):
        img_path = os.path.join(self.dir_imgs, self.points.iloc[item, 0])
        img_nose = eval(self.points.iloc[item, 1])

        img = Image.open(img_path).convert('RGB')

        x, y = map(int, img_nose)
        img_size = img.size
        new_x = x * (256 / img_size[0])
        new_y = y * (256 / img_size[1])
        new_x = (new_x / 256) * 2 - 1
        new_y = (new_y / 256) * 2 - 1

        nose_coordinates = torch.tensor(data=[new_x, new_y], dtype=torch.float32)

        if self.transform:
            data = {"image": img, "points": img_nose}
            data = self.transform(data)
            img, img_nose = data["image"], data["points"]

        return img_path, img, nose_coordinates


class resize_with_points(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image = data["image"]
        points = data["points"]

        resize_image = functional.resize(image, self.size)
        resize_image = functional.to_tensor(resize_image)

        resize_points = [p * self.size[i] / image.size[i] for i, p in enumerate(points)]

        return {"image": resize_image, "points": resize_points}


def train(num_epochs, train_loader, model, loss_fn, optimizer, scheduler, save_model, plot_name,
          device):
    print("training... ")

    losses = []
    actual_point = []
    prediction_point = []

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.strftime("%H:%M:%S %p")
        print(f'Starting Epoch {epoch + 1} at {epoch_start_time}')
        epoch_start_time = time.time()
        epoch_loss = 0.0
        i = 1
        for _, (img_path, imgs, points) in enumerate(train_loader):
            imgs = imgs.to(device=device)
            points = points.to(device=device)

            outputs = model(imgs)
            loss = loss_fn(outputs, points)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            prediction_point.extend(outputs.detach().cpu().numpy())
            actual_point.extend(points.detach().cpu().numpy())

            if i == 5 and epoch % 2 == 0:
                img_name = img_path[3]
                pred_points = outputs.detach().cpu().numpy()
                predicted_nose = pred_points[3]
                display_image(img_name=img_name, predicted_nose=predicted_nose)

            i += 1

        losses.append(epoch_loss / len(train_loader))

        scheduler.step()

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        print(f'Epoch {epoch + 1} took {epoch_time:.2f} seconds -- about {int(epoch_time / 60)} minutes')

    model.to("cpu")
    torch.save(model.state_dict(), f'{save_model}_e{num_epochs}.pth')
    print(f'Model trained and saved to {save_model}')

    end_time = time.time()
    total_training_time = end_time - start_time
    print(f'Training Time: {total_training_time} seconds -- about {int(total_training_time / 60)} minutes')

    plt.figure()
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curve')
    plt.savefig(f'outputs/training/loss_plots/{plot_name}_e{num_epochs}.png')
    plt.show()


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()

    argParser.add_argument('-t', '--input_dir_txt', type=str, help='input dir (./)', default='./')
    argParser.add_argument('-i', '--input_dir_img', type=str, help='input dir (./)', default='./')
    argParser.add_argument('-e', '--epochs', type=int, default=40, help='number of epochs')
    argParser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
    argParser.add_argument('-s', '--save', type=str, default='train', help='save model path')
    argParser.add_argument('-p', '--plot', type=str, default='train_loss_plot', help='save plot name')

    args = argParser.parse_args()

    input_dir_txt = args.input_dir_txt
    input_dir_img = args.input_dir_img
    num_epochs = args.epochs
    batch_size = args.batch_size
    save_model = args.save
    plot_name = args.plot

    print(f"PyTorch version: {torch.__version__}")
    print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
    print(f"Is MPS available? {torch.backends.mps.is_available()}")
    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        print('using cuda ...')
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif torch.backends.mps.is_available():
        print('using mps ...')
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        print('using cpu ...')
        device = torch.device("cpu")
    # device = torch.device("cpu")
    # device = torch.device("mps")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        resize_with_points((256, 256))
    ])

    training_set = nose_dataset(dir_txt=input_dir_txt, dir_imgs=input_dir_img, transform=transform)

    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)

    model = model.NoseRegressor()

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.05)

    model.train()
    model.to(device=device)

    train(num_epochs=num_epochs, train_loader=train_loader, model=model, loss_fn=loss_fn,
          optimizer=optimizer, scheduler=scheduler, save_model=save_model, plot_name=plot_name,
          device=device)

    print("training completed")
    exit(0)
