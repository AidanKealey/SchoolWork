import datetime
import torch
from torch import optim as optim
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import argparse
from model import autoencoderMLP4Layer # import autoencoder

device = "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device)
print(f"Using device: {device}")

# get input from command line
argParser = argparse.ArgumentParser()

argParser.add_argument("-z", "--bottleneck_size", type=int, default=8, help="bottleneck size")
argParser.add_argument("-e", "--epochs", type=int, default=50, help="number of epochs")
argParser.add_argument("-b", "--batch_size", type=int, default=2048, help="batch size")
argParser.add_argument("-s", "--save", type=str, default="MLP.8.pth", help="save model")
argParser.add_argument("-p", "--plot",  type=str, default="loss.MLP.8.png", help="save plot")


command_input = argParser.parse_args()

# Data transformations
train_transform = transforms.Compose([transforms.ToTensor()]) # transforming the images to tensors
train_set = MNIST("./data/mnist", train=True, download=True, transform=train_transform)
train_loader = DataLoader(train_set, batch_size=command_input.batch_size, shuffle=True)

# instantiate autoencoder
myModel = autoencoderMLP4Layer(N_bottleneck=command_input.bottleneck_size)

# loss and optimizer
loss_function = torch.nn.MSELoss()
optimizer = optim.Adam(params=myModel.parameters(), lr=1e-3, weight_decay=1e-5)

#Training

def train(n_epochs, optimizer, model, loss_fn, train_loader, device):
    print('training ...')
    model.train()
    losses_train = []

    for epoch in range(1, n_epochs + 1):
        print('epoch ', epoch)
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = torch.flatten(imgs, start_dim=1)
            imgs = imgs.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        losses_train += [loss_train / len(train_loader)]

        print('{}, Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch, loss_train / len(train_loader)
        ))

    # Before saving the model, print the filename
    print("Saving model as:", command_input.save)

    return losses_train


# Train the model
device = torch.device("cpu")
myModel.to(device)
losses = train(command_input.epochs, optimizer, myModel, loss_function, train_loader, device)

# Save the trained model
torch.save(myModel.state_dict(), command_input.save)

# Plot and save the loss curve
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.savefig("loss_MLP_8.png")
plt.show()





