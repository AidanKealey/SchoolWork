import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import argparse

class ModifiedDecoder(nn.Module):
    def __init__(self):
        super(ModifiedDecoder, self).__init__()

        self.residual_block1 = self._make_residual_block(512, 256)
        self.residual_block2 = self._make_residual_block(256, 256)

        self.final_conv = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(256 * 4 * 4, 100)

    def _make_residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        residual = x  # Save the input for the skip connection
        x = self.residual_block1(x)

        # Check if the dimensions match before adding the skip connection
        if x.size(1) == residual.size(1):  # Ensure matching number of channels
            x = x + residual

        x = self.residual_block2(x)
        x = self.final_conv(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

class encoder_decoder:
    # Built off of Lab 2
    encoder = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
    )
    decoder = ModifiedDecoder()

class network(nn.Module):
    # Built off of Lab 2
    def __init__(self, encoder, decoder=None):
        super(network, self).__init__()
        self.encoder = encoder
        # freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decoder = decoder

        #   if no decoder loaded, then initialize with random weights
        if self.decoder == None:
            # self.decoder = _decoder
            self.decoder = encoder_decoder.decoder
            self.init_decoder_weights(mean=0.0, std=0.01)

    def init_decoder_weights(self, mean, std):
        for param in self.decoder.parameters():
            nn.init.normal_(param, mean=mean, std=std)

    def encode(self, x):
        return (self.encoder(x))

    def decode(self, X):
        return(self.decoder(X))

    def forward(self, content):
        encode = self.encode(content)

        return self.decode(encode)


def train(num_epochs, model, criterion, optimizer, scheduler, train_loader, device):
    # Lists to store training stats
    losses = []
    top1_errors = []
    top5_errors = []

    start_time = time.time()
    for epoch in range(num_epochs):
        print(f'Starting Epoch {epoch + 1}')
        total_num_samples = 0
        top1_correct = 0
        top5_correct = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate top-1 and top-5 correct
            _, predicted = torch.max(outputs.data, 1)
            total_num_samples += labels.size(0)
            top1_correct += (predicted == labels).sum().item()
            _, predicted_top5 = torch.topk(outputs.data, 5, dim=1)
            top5_correct += (torch.sum(predicted_top5 == labels.view(-1, 1))).item()

        accuracy = (top1_correct / total_num_samples) * 100

        top1_error = 1 - top1_correct / total_num_samples
        top5_error = 1 - top5_correct / total_num_samples

        losses.append(loss.item())
        top1_errors.append(top1_error)
        top5_errors.append(top5_error)
        scheduler.step()

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Top-1 Error: {top1_error:.2%}, Top-5 Error: {top5_error:.2%}, Accuracy: {accuracy:.2f}%'
        )

    model.to("cpu")
    torch.save(model.decoder.state_dict(), args.save_path)

    end_time = time.time()
    total_training_time = end_time - start_time
    print(f'Training Time: {total_training_time} seconds -- about {int(total_training_time / 60)} minutes')

    # Plot the loss and errors
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')

    plt.subplot(1, 3, 2)
    plt.plot(top1_errors)
    plt.xlabel('Epoch')
    plt.ylabel('Top-1 Error %')
    plt.title('Top-1 Error over Epochs')

    plt.subplot(1, 3, 3)
    plt.plot(top5_errors)
    plt.xlabel('Epoch')
    plt.ylabel('Top-5 Error %')
    plt.title('Top-5 Error over Epochs')

    plt.tight_layout()
    plt.savefig(args.plot_path)
    plt.show()

    print("Finished Training")


if __name__ == '__main__':
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
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description='Training script for Vanilla Model')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-en', '--encoder', type=str, default='encoder.pth', help='Path to encoder')
    parser.add_argument('-s', '--save_path', type=str, default='vanilla.pth', help='Path to save trained model')
    parser.add_argument('-p', '--plot_path', type=str, default='training_plot.png', help='Path to save training plot')
    args = parser.parse_args()

    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    cifar100_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=None)

    mean = torch.tensor(cifar100_train.data.mean(axis=(0, 1, 2)) / 255.0)
    std = torch.tensor(cifar100_train.data.std(axis=(0, 1, 2)) / 255.0)

    # Data transformations with normalized mean and std
    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Apply the transformations to the dataset
    cifar100_train.transform = cifar_transform

    train_loader = DataLoader(cifar100_train, batch_size=batch_size, shuffle=True)

    encoder = encoder_decoder.encoder
    encoder.load_state_dict(torch.load(args.encoder, map_location='cpu'))
    decoder = encoder_decoder.decoder
    model = network(encoder=encoder, decoder=decoder).to(device)

    model.train()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train(num_epochs=num_epochs, model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, train_loader=train_loader, device=device)

    exit(0)