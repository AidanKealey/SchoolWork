import torch
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from model import autoencoderMLP4Layer
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Function to get user-selected image index
# Ask for 2 inputs (use one for the interpolation)
def getIdx():
    print("To select an MNIST image please")
    value = input("Enter an integer between 0 and 5999 and click enter: ")
    value2= input("Enter an integer between 0 and 5999 and click enter: ")
    return int(value), int(value2)


# add noise to the image
def addNoise(image):
    print("adding noise to image ...")
    noise_level = 0.5
    noise = torch.rand(image.size(), dtype=image.dtype, layout=image.layout, device=image.device) * noise_level
    noisy_image = image + noise
    noisy_image = torch.clamp(noisy_image, min=0.0, max=1.0)

    return noisy_image


# linear interpolation between two images
def linear_interpolation(image1, image2, weight):
    print("Doing linear interpolation...")
    return torch.lerp(image1, image2, weight)


# Step 2: display image from MNIST set
train_transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.MNIST("./data/mnist", train=True, download=True, transform=train_transform)
idx, idx2 = getIdx()
original_image, _ = train_set[idx]
original_image2, _ = train_set[idx2]
plt.imshow(original_image.squeeze().numpy(), cmap='gray')  # removes the channel dimension so it's just 28x28
plt.title("Original Image")
plt.show()

# Step 3: training the model


# Step 4: Load the trained model
# Instantiate the model
my_model = autoencoderMLP4Layer(N_bottleneck=8)
my_model.load_state_dict(torch.load("MLP.8.pth"))
my_model.eval()  # Set the model to evaluation mode

# format image
input_image = torch.flatten(original_image).float()
input_image /= input_image.max()

# put image through autoencoder
with torch.no_grad():  # Disable gradient calculations
    reconstructed_image = my_model(input_image)

# reshape input and reconstructed images
input_image = torch.reshape(torch.flatten(input_image), (1, 28, 28))
reconstructed_image = torch.reshape(torch.flatten(reconstructed_image), (1, 28, 28))

# Plot the results
f = plt.figure()
f.add_subplot(1, 2, 1)
plt.imshow(input_image[0], cmap='gray')
f.add_subplot(1, 2, 2)
plt.imshow(reconstructed_image[0], cmap='gray')
plt.show()


# Step 5: Image Denoising
# Add noise to the original image
noisy_image = addNoise(original_image)

# Format images for autoencoder input using reshape
input_image = torch.reshape(noisy_image, (1, -1)).float()  # 1x784 tensor
input_image /= input_image.max()  # Normalize to [0,1]


# Pass noisy image through the autoencoder
with torch.no_grad():  # Disable gradient calculations
    reconstructed_image = my_model(input_image)

# Reshape input_image, noisy_image, and reconstructed_image for plotting
input_image = noisy_image.squeeze().numpy()
noisy_image = noisy_image.squeeze().numpy()
reconstructed_image = reconstructed_image.view(28, 28).numpy()

# Plot the original, noisy, and reconstructed images side-by-side
f, axarr = plt.subplots(1, 3)
axarr[0].imshow(original_image.squeeze().numpy(), cmap='gray')
axarr[1].imshow(input_image, cmap='gray')
axarr[2].imshow(reconstructed_image, cmap='gray')
plt.show()


# Step 6: Linear Interpolation between Noisy and Reconstructed Images

# Instantiate the model
my_model = autoencoderMLP4Layer(N_bottleneck=8)
my_model.load_state_dict(torch.load("MLP.8.pth"))
my_model.eval()  # Set the model to evaluation mode

# Format images for autoencoder input
# Flatten and normalize the input image
input_image1 = original_image.view(1, -1).float()  # Reshape to (1, 784)
input_image1 /= input_image1.max()  # Normalize to [0,1]

# Reshape input_image2 and normalize
input_image2 = original_image2.view(1, -1).float()  # Reshape to (1, 784)
input_image2 /= input_image2.max()  # Normalize to [0,1]

# Pass images through autoencoder to get bottleneck tensors
with torch.no_grad():  # Disable gradient calculations
    bottleneck1 = my_model.encode(input_image1)
    bottleneck2 = my_model.encode(input_image2)

    # Number of interpolation steps
    steps = 8

    # Linear interpolation between bottleneck tensors and decode them
    interpolated_images = []
    weight_values = torch.linspace(0, 1, steps)

    for weight in weight_values:
        interpolated_bottleneck = linear_interpolation(bottleneck1, bottleneck2, weight)
        reconstructed_image = my_model.decode(interpolated_bottleneck)
        interpolated_images.append(reconstructed_image)

    plt.figure(figsize=(8, 3))

    for i in range(steps):
        interpolated_image = interpolated_images[i].view(28, 28).numpy()
        plt.subplot(1, steps + 2, i + 1)  # Subplots for interpolated images
        plt.imshow(interpolated_image, cmap='gray')

    plt.show()


