# Python file to perform various computation on MNIST digits passed through an MLP autoencoder
# Written by Patrick Lister 20161956 and Deven Shidfar _______, September 19, 2023

import torch
from mnist_model import mnistautoencoder
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
import sys
import getopt

def reconstruction(state_dict='euclid_v0.pth'):
    #Instatiate model
    model = mnistautoencoder()
    model.load_state_dict(torch.load(f"./saved_train/{state_dict}"))
    model.eval()

    #Get eval data
    train_transform = transforms.Compose([transforms.ToTensor()])
    eval_set = MNIST('./data/mnist', train=False, download=True,
    transform=train_transform)

    #Get image to test
    idx = int(input("Please input an integer: "))
    image, label = eval_set[idx]

    #Put image through model
    with torch.no_grad():
        output = model(image.view(1,-1))

    #Reshape input and output to squares for plotting
    output = output.reshape(28,28)
    image = image.reshape(28,28)

    #Plot results
    f = plt.figure()
    f.add_subplot(1,2,1)
    plt.imshow(image, cmap='gray')
    f.add_subplot(1,2,2)
    plt.imshow(output, cmap='gray')
    plt.show()

def denoise(state_dict='MLP.8.pth'):
    '''Denoising section'''
    #Instatiate model
    model = mnistautoencoder()
    model.load_state_dict(torch.load(f"./saved_train/{state_dict}"))
    model.eval()

    #Get eval data
    train_transform = transforms.Compose([transforms.ToTensor()])
    eval_set = MNIST('./data/mnist', train=False, download=True,
    transform=train_transform)

    #Get image to test on
    idx = int(input("Please input an integer: "))
    image, label = eval_set[idx]

    #Add noise to input image
    noise = torch.rand(image.shape) - 0.5
    noisy_input = image + noise * 1

    #Put noisy image through model
    with torch.no_grad():
        output = model(noisy_input.view(1,-1))

    #Reshape images for plotting
    output = output.reshape(28,28)
    image = image.reshape(28,28)
    noisy_input = noisy_input.reshape(28,28)

    #Plot results
    f = plt.figure()
    f.add_subplot(1,3,1)
    plt.imshow(image, cmap='gray')
    f.add_subplot(1,3,2)
    plt.imshow(noisy_input, cmap='gray')
    f.add_subplot(1,3,3)
    plt.imshow(output, cmap='gray')
    plt.show()

def interpolate(state_dict='MLP.8.pth'):
    '''Interpolation section'''
    #Instatiate model
    model = mnistautoencoder()
    model.load_state_dict(torch.load(f"./saved_train/{state_dict}"))
    model.eval()

    # Define alpha values
    div = 10
    alphas = [i/div for i in range(div+1)]

    #Get eval portion of mnist data set
    train_transform = transforms.Compose([transforms.ToTensor()])
    eval_set = MNIST('./data/mnist', train=False, download=True,
    transform=train_transform)

    #Get images to interpolate
    idx1, idx2 = input("Please input two integers: ").split()
    idx1 = int(idx1)
    idx2 = int(idx2)
    image1, label1 = eval_set[idx1]
    image2, label2 = eval_set[idx2]

    #Encode to bottleneck
    with torch.no_grad():
        output1 = model.encode(image1.view(1,-1))
        output2 = model.encode(image2.view(1,-1))

    #Interpolate the bottleneck tensors
    interpolated = []
    for alpha in alphas:
        interp = torch.lerp(output1, output2, alpha)
        interpolated.append(interp)

    #Decode interpolated bottleneck tensors
    interpolated_out = []
    with torch.no_grad():
        for interp in interpolated:
            interp = model.decode(interp)
            interpolated_out.append(interp)

    #Reshape the interpolated tensors to images
    images_out = []
    for interp in interpolated_out:
        interp = interp.reshape(28,28)
        images_out.append(interp)

    #Reshape input images to squares if needed
    image1 = image1.reshape(28,28)
    image2 = image2.reshape(28,28)

    #Plot results
    f = plt.figure()
    size = len(alphas)
    for i in range(1,size):
        f.add_subplot(1,size,i)
        plt.imshow(images_out[i-1], cmap='gray')

    plt.show()

    test = 1

state_dict = None

#Get command line arguments
argv = sys.argv[1:]
try:
    options, args = getopt.getopt(argv, "l:",
                            ["first =",
                            "last ="])
except:
    print("Not a recognized flag")

for name, value in options:
    if name in ['-l']:
        state_dict = value


#Call functions for each desired output
reconstruction(state_dict)
reconstruction(state_dict)
reconstruction(state_dict)
denoise(state_dict)
interpolate(state_dict)
