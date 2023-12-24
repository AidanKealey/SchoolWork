import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train_mod import network, encoder_decoder
import argparse


# Function to test the model
def test_model(model, test_loader, device):
    print("testing...")

    total_num_samples = 0
    top1_correct = 0
    top5_correct = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)

            _, predicted = torch.max(outputs.data, 1)
            total_num_samples += labels.size(0)
            top1_correct += (predicted == labels).sum().item()
            _, predicted_top5 = torch.topk(outputs.data, 5, dim=1)
            top5_correct += (torch.sum(predicted_top5 == labels.view(-1, 1))).item()

    accuracy = (top1_correct / total_num_samples) * 100

    top1_error = 1 - top1_correct / total_num_samples
    top5_error = 1 - top5_correct / total_num_samples

    print(
        f'Top-1 Error: {top1_error:.2%}, Top-5 Error: {top5_error:.2%}, Accuracy: {accuracy:.2f}%'
    )

    print("Finished Testing")


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

    parser = argparse.ArgumentParser(description='Testing parameters for Vanilla Model')
    parser.add_argument('-en', '--encoder', type=str, default='encoder.pth', help='Path to encoder')
    parser.add_argument('-de', '--decoder', type=str, default='vanilla.pth', help='Path to decoder')
    args = parser.parse_args()

    encoder_name = args.encoder
    decoder_name = args.decoder

    cifar100_test = datasets.CIFAR100(root='./data', train=False, download=True, transform=None)

    mean = torch.tensor(cifar100_test.data.mean(axis=(0, 1, 2)) / 255.0)
    std = torch.tensor(cifar100_test.data.std(axis=(0, 1, 2)) / 255.0)

    # Data transformations with normalized mean and std
    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    cifar100_test.transform = cifar_transform

    test_loader = DataLoader(cifar100_test, batch_size=100, shuffle=True)

    encoder = encoder_decoder.encoder
    encoder.load_state_dict(torch.load(encoder_name, map_location='cpu'))
    decoder = encoder_decoder.decoder
    decoder.load_state_dict(torch.load(decoder_name, map_location='cpu'))

    model = network(encoder=encoder, decoder=decoder).to(device)

    model.to(device=device)
    model.eval()

    # Test the model
    test_model(model=model, test_loader=test_loader, device=device)

    exit(0)