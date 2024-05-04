import torch.optim as opt
from torch.utils.data import DataLoader
import functions
import torch
import mnist_model
from torchvision.datasets import MNIST
from torchvision import transforms

# Parse arguments
batch_size, n_epochs, weight_file, loss_image, device, weight_decay = functions.parse_args()

# Prepare dataloaders
train_transform = transforms.Compose([transforms.ToTensor(), torch.flatten])
test_transform = transforms.Compose([transforms.ToTensor(), torch.flatten])
train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)
validation_set = MNIST('./data/mnist', train=False, download=True, transform=train_transform)
train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=512, shuffle=True)

# Instantiate Model
model = mnist_model.mnistautoencoder()
model.to(device)

# Pepare optimizer, scheduler, and loss function
optimizer = opt.Adam(model.parameters(), lr=1e-3)
scheduler = opt.lr_scheduler.ExponentialLR(optimizer, gamma=1)
loss_fn=torch.nn.BCELoss(reduction='sum')

functions.train(model, n_epochs, train_loader, val_loader, device, optimizer, scheduler, loss_fn=loss_fn, loss_file=loss_image, autoencoder=True, weight_file=weight_file)