import os
import torch
import time
import argparse
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

class KittiTrainDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir
        self.transform = transform

        with open(os.path.join(dir, 'labels.txt'), 'r') as file:
            lines = file.readlines()
            self.img_files = [os.path.join(dir, line.split()[0]) for line in lines]
            self.labels = [int(line.split()[1]) for line in lines]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def train_yoda_classifier(num_epochs, train_loader, model, loss_fn, optimizer, scheduler, save_path, plot_path,
                          accuracy_path, device):
    losses = []
    actual_label = []
    prediction_label = []

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.strftime("%H:%M:%S %p")
        print(f'Starting Epoch {epoch + 1} at {epoch_start_time}')
        epoch_start_time = time.time()
        epoch_loss = 0.0
        i = 1
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            _, predictions = torch.max(outputs, 1)
            actual_label.extend(labels.cpu().numpy())
            prediction_label.extend(predictions.cpu().numpy())

            if (i % 100) == 0:
                print(f"Completed epoch: {epoch + 1}, batch: {i}")
            i += 1

        losses.append(epoch_loss / len(train_loader))

        scheduler.step()

        epoch_end_time = time.time()  # Record the end time of the epoch
        epoch_time = epoch_end_time - epoch_start_time
        print(f'Epoch {epoch + 1} took {epoch_time:.2f} seconds -- about {int(epoch_time / 60)} minutes')

    # save trained model
    model.to("cpu")
    torch.save(model.state_dict(), save_path)
    print(f'Model trained and saved to {save_path}')

    end_time = time.time()
    total_training_time = end_time - start_time
    print(f'Training Time: {total_training_time} seconds -- about {int(total_training_time / 60)} minutes')

    # plot the loss curve and confusion matrix
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curve')

    plt.subplot(1, 2, 2)
    cm = confusion_matrix(actual_label, prediction_label)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
    cm_display.plot()
    plt.title('Confusion Matrix')

    plt.savefig(plot_path)

    plt.show()

    print(f'Loss plot and Confusion matrix saved to {plot_path}')

    acc = accuracy_score(actual_label, prediction_label)
    with open(accuracy_path, 'w') as file:
        file.write(f'Accuracy: {acc}')
    print(f'Accuracy saved to {accuracy_path}')


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-i', '--input_dir', type=str, help='input dir (./)', default='./')
    argParser.add_argument('-e', '--epochs', type=int, default=40, help='number of epochs')
    argParser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
    argParser.add_argument('-s', '--save', type=str, default='step3_train_yoda_classifier.pth', help='save model path')
    argParser.add_argument('-p', '--plot', type=str, default='step3_train_loss_plot.png', help='save plot path')
    argParser.add_argument('-a', '--accuracy', type=str, default='step3_train_accuracy.txt', help='save accuracy path')

    args = argParser.parse_args()

    input_dir = args.input_dir
    num_epochs = args.epochs
    batch_size = args.batch_size
    save_path = args.save
    plot_path = args.plot
    accuracy_path = args.accuracy

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
    device = torch.device(device)
    torch.device(device)
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    ROI_train_set = KittiTrainDataset(dir=os.path.join(input_dir, 'train'), transform=transform)

    train_loader = DataLoader(ROI_train_set, batch_size=batch_size, shuffle=True)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)

    loss_fn = nn.CrossEntropyLoss().to(device=device)
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    model.train()
    model.to(device=device)

    train_yoda_classifier(num_epochs=num_epochs, train_loader=train_loader, model=model, loss_fn=loss_fn,
                          optimizer=optimizer, scheduler=scheduler, save_path=save_path, plot_path=plot_path,
                          accuracy_path=accuracy_path, device=device)

    print("training completed")
    exit(0)