import argparse
import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data import Dataset
from PIL import Image, ImageFile

def parse_args():
    '''Function to parse arguments for a general training function'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', type=int, help='epochs')
    parser.add_argument('-b', type=int, help='batch size')
    parser.add_argument('-w', type=int, help='weight decay')
    parser.add_argument('-l', type=str, help='weight file without .pth, will be appended during training')
    parser.add_argument('-p', type=str, help='Loss output image, full path')
    parser.add_argument('-cuda', type=str, help='[Y/N] for gpu usage')
    opt = parser.parse_args()
    batch_size = opt.b
    epochs = opt.e
    weight_decay = opt.w
    weight_file = opt.l
    loss_image = opt.p
    if (opt.cuda == 'y' or opt.cuda == 'Y') and (torch.cuda.is_available()):
        use_cuda = True
        device = 'cuda'
    else:
        device = 'cpu'

    return batch_size, epochs, weight_file, loss_image, device, weight_decay

class custom_dataset(Dataset):
    '''Dataset class for custom datasets'''
    def __init__(self, dir, label_file, transform=None):
        super().__init__()
        Image.MAX_IMAGE_PIXELS = None
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.transform = transform
        self.image_files = [dir + file_name for file_name in os.listdir(dir)]
        self.labels = self.get_labels(label_file)
        self.image_files = self.image_files[:-2]

    def get_labels(self, label_file):
        f = open(label_file)
        lines = f.readlines()
        return lines

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = Image.open(self.image_files[index]).convert('RGB')
        image_sample = self.transform(image)
        label = int(self.labels[index].split(' ')[1])
        # print('break 27: ', index, image, image_sample.shape)
        return image_sample, label

# Code adapted from https://colab.research.google.com/drive/1TrhEfI3stJ-yNp7_ZxUAtfWjj-Qe_Hym?usp=sharing
def accuracy_loss(model, dl, device, loss_fn):
    '''Function to calculate the accuracy and loss for a classification model'''
    model.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for data in dl:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            vals, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += loss_fn(outputs, labels).cpu().item() / len(dl)
    return 100*correct/total, loss

def loss_autoencoder(model, dl, device, loss_fn):
    '''Function to calculate the loss for a model'''
    model.eval()
    loss = 0
    with torch.no_grad():
        for data in dl:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += loss_fn(outputs, inputs).cpu().item() / len(dl)
    return loss

def train(model, n_epochs, train_dl, test_dl, device, optimizer, scheduler, loss_fn, loss_file, weight_file, autoencoder=False):
    '''Function to train autoencoder and classifier'''
    model.train()
    print("Training...")
    losses_list = []
    test_accuracy = []
    test_loss = []
    for epoch in range(n_epochs):
        loss_epoch = 0.0
        index = 0

        # Iterate over batches
        for imgs, labels in train_dl:
            index += 1
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(imgs)

            if autoencoder == True:
                loss = loss_fn(output, imgs) # Loss calculated on input and recreation
            else:
                loss = loss_fn(output, labels) # Loss calculated on prediction and ground truth

            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()
        
        # Finished epoch now validate on test loader
        if autoencoder == True:
            test_l = loss_autoencoder(model, test_dl, device, loss_fn)
            print(f"Test loss epoch {epoch}: {test_l}")
            test_loss.append(test_l)
        else:
            test_a, test_l = accuracy_loss(model, test_dl, device, loss_fn)
            print(f"Test accuracy epoch {epoch}: {test_a}")
            test_accuracy.append(test_a)
            test_loss.append(test_l)

        # Print loss for each epoch
        losses_list.append(loss_epoch / index)
        print(f"Epoch {epoch} loss: {losses_list[epoch]}")

        # Periodically save weight files
        if epoch % 10 == 0 or epoch == n_epochs:
            state_dict = model.state_dict()
            torch.save(state_dict, f"{weight_file}_{epoch}.pth")

        model.train() # Call after testing
        scheduler.step() 
        
    # Plot loss graph and save to .png
    plt.plot(losses_list, label="Train Loss")
    plt.plot(test_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.savefig(loss_file)