import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as opt
import torch
import datetime as dt
import mnist_model
import prodigyopt as prodigy

torch.cuda.empty_cache()

n_epochs = 25

train_transform = transforms.Compose([transforms.ToTensor()])

train_set = MNIST('./data/mnist', train=True, download=True,
transform=train_transform)
validation_set = MNIST('./data/mnist', train=False, download=True,
transform=train_transform)

device = 'cuda'

model = mnist_model.mnistautoencoder()
model.to(device)

train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=512, shuffle=True)

optimizer = opt.Adam(model.parameters(), lr=1e-3)
scheduler = opt.lr_scheduler.ExponentialLR(optimizer, gamma=1)
#optimizer = prodigy.Prodigy(model.parameters(), lr=1., weight_decay=0.9, decouple=True, safeguard_warmup=True)
#scheduler = opt.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

loss_fn = nn.MSELoss(reduction='sum')

def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])

#val_loader = False

print('Training...')
model.train() #!!!!
losses_train = [] 
losses_val = []
for epoch in range(1, n_epochs+1):
    print('epoch ', epoch)
    # Run a validation set through current weights
    if (val_loader != False):
        for imgs_val, labels in val_loader:
            imgs_val = imgs_val.view(imgs_val.size(0), -1)
            loss_val = 0.0
            imgs_val = imgs_val.to(device)
            with torch.no_grad():
                outputs_val = model(imgs_val)
            loss_v = loss_fn(outputs_val, imgs_val)
            loss_val += loss_v.item()
            continue
    # Run training data through model and update weights
    batch_num = 0
    loss_train = 0.0
    for imgs, labels in train_loader:
        model.train()
        batch_num += 1
        imgs = imgs.view(imgs.size(0), -1)
        imgs = imgs.to(device=device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = loss_fn(outputs, imgs)
        #test = loss.grad_fn
        loss.backward(retain_graph=False)
        #test = getBack(loss.grad_fn)
        optimizer.step()
        loss_train += loss.item()
        if batch_num % 5000 == 0:
            print(f'Batch: {batch_num}')
            print(f"Loss: {loss}")
    scheduler.step()
    losses_train += [loss_train/len(train_loader)]
    losses_val += [loss_val/len(val_loader)]
    if epoch % 5 == 0:
        state_file = f'euclid_squared_autoencoder_v2_epoch_{epoch}.pth'
        torch.save(model.state_dict(), f'./saved_train/{state_file}')

    print('{} Epoch {}, Training loss {}'.format(dt.datetime.now(), epoch, loss_train/len(train_loader)))

state_file = 'euclid_squared_autoencoder_v2.pth'
torch.save(model.state_dict(), f'./saved_train/{state_file}')

loss_graph = 'loss_euclid_squared_autoencoder_v2'
plt.plot(losses_train,label='Training Loss')
plt.plot(losses_val, label='Validation Loss')
plt.savefig(f"./loss_graph/{loss_graph}")
