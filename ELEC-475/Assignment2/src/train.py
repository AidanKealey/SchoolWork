import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from custom_dataset import custom_dataset
from AdaIN_net import AdaIN_net, encoder_decoder
import torch
import matplotlib.pyplot as plt
import datetime

def train(args, device):
    print('training...')

    content_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    # resize the datasets
    content_data = custom_dataset(args.content_dir, transform=content_transform)
    style_data = custom_dataset(args.style_dir, transform=style_transform)

    # dataloaders
    content_loader = DataLoader(content_data, batch_size=args.batch_size, shuffle=True)
    style_loader = DataLoader(style_data, batch_size=args.batch_size, shuffle=True)

    # initialize model
    decoder = encoder_decoder.decoder.to(device)
    encoder = encoder_decoder.encoder
    encoder.load_state_dict(torch.load(args.encoder_weights, map_location='cpu'))
    encoder = encoder.to(device)
    model = AdaIN_net(encoder, decoder).to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Lists to store losses
    content_losses = []
    style_losses = []
    total_losses = []

    # Set model to train
    model.train()
    model.to(device)

    for epoch in range(args.epochs):
        content_epoch_loss = 0.0
        style_epoch_loss = 0.0
        total_epoch_loss = 0.0
        epoch_start_time = datetime.datetime.now()

        # loop through 2 datasets simultaneously
        for content_batch, style_batch in zip(content_loader, style_loader):
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            # Forward pass
            loss_content, loss_style = model(content_batch, style_batch)

            # Calculate the total loss as a combination of content and style losses
            total_loss = loss_content + args.gamma * loss_style

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # append losses for each batch
            content_epoch_loss += loss_content.item()
            style_epoch_loss += loss_style.item()
            total_epoch_loss += total_loss.item()

        # average losses for the epoch
        content_epoch_loss /= len(content_data)
        style_epoch_loss /= len(style_data)
        total_epoch_loss /= len(content_data)

        # append epoch losses to the lists
        content_losses.append(content_epoch_loss)
        style_losses.append(style_epoch_loss)
        total_losses.append(total_epoch_loss)

        print(
            f"Epoch {epoch + 1}/{args.epochs}, Content Loss: {content_epoch_loss}, Style Loss: {style_epoch_loss}, "
            f"Total Loss: {total_epoch_loss}"
        )

        epoch_end_time = datetime.datetime.now()
        delta_epoch_time_from_start = epoch_end_time - start_time
        delta_epoch_time = epoch_end_time - epoch_start_time

        print(f"Epoch Start Time: {epoch_start_time.time()} \nEpoch End Time: {epoch_end_time.time()} \n"
              f"Epoch Delta Time: {delta_epoch_time} \n"
              f"Delta Time From Start: {delta_epoch_time_from_start}")

    training_end_time = datetime.datetime.now()
    print(f"Training end time: {training_end_time.time()}")
    delta_training_time = training_end_time - start_time
    print(f"Training total time: {delta_training_time}")

    # Plot losses
    plt.plot(content_losses, label='Content Loss')
    plt.plot(style_losses, label='Style Loss')
    plt.plot(total_losses, label='Total Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(args.output_path)
    plt.show()

    # save model
    model.to("cpu")
    torch.save(model.decoder.state_dict(), args.decoder_weights)

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("-content_dir", type=str, help="Path to the content_dir dataset.")
arg_parser.add_argument("-style_dir", type=str, help="Path to the style_dir dataset.")
arg_parser.add_argument("-gamma", type=float, default=1.0, help="Gamma parameter.")
arg_parser.add_argument("-e", "--epochs", type=int, default=20, help="Number of epochs.")
arg_parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size.")
arg_parser.add_argument("-l", "--encoder_weights", type=str, help="Path to the encoder weights.")
arg_parser.add_argument("-s", "--decoder_weights", type=str, help="Path to the decoder weights.")
arg_parser.add_argument("-p", "--output_path", type=str, help="Path to the output image.")

args = arg_parser.parse_args()

print(f"Content Directory: {args.content_dir}, Style Directory: {args.style_dir}\n",
      f"Gamma: {args.gamma}, Number of Epochs: {args.epochs}, Batch Size: {args.batch_size}\n",
      f"Load Encoder Weights: {args.encoder_weights}, Create Decoder Weights: {args.decoder_weights}, ",
      f"Loss Plot: {args.output_path}")

print(f"PyTorch version: {torch.__version__}")

print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")
device = "mps" if torch.backends.mps.is_available() else "cpu"
# device = torch.device("cpu")
# device = torch.device("mps")
device = torch.device(device)
print(f"Using device: {device}")

start_time = datetime.datetime.now()
print(f"Training start time: {start_time.time()}")

train(args, device)
