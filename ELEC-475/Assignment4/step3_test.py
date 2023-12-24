import os
import torch
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

class KittiTestDataset(Dataset):
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


def test_yoda_classifier(model, test_loader, confusion_matrix_path, accuracy_path, device):
    actual_label = []
    prediction_label = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            outputs = model(imgs)

            _, predictions = torch.max(outputs, 1)
            actual_label.extend(labels.cpu().numpy())
            prediction_label.extend(predictions.cpu().numpy())

    plt.figure(figsize=(12, 4))
    cm = confusion_matrix(actual_label, prediction_label)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
    cm_display.plot()
    plt.title('Confusion Matrix')
    plt.savefig(confusion_matrix_path)
    plt.show()

    acc = accuracy_score(actual_label, prediction_label)
    print(f'Test Accuracy: {acc}')
    with open(accuracy_path, 'w') as file:
        file.write(f'Test Accuracy: {acc}')
    print(f'Accuracy saved to {accuracy_path}')


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-i', '--input_dir', type=str, help='input dir (./)', default='./')
    argParser.add_argument('-m', '--model', type=str, default='yoda_classifier_trained.pth', help='model name')
    argParser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
    argParser.add_argument('-c', '--confusion_matrix', type=str, default='step3_test_confusion_matrix.png',
                           help='save confusion matrix path')
    argParser.add_argument('-a', '--accuracy', type=str, default='step3_test_accuracy.txt', help='save accuracy path')

    args = argParser.parse_args()

    input_dir = args.input_dir                  # 'data/Kitti8_ROIs'
    batch_size = args.batch_size                # 100
    model_name = args.model                     # 'yoda_classifier_train_e25.pth'
    confusion_matrix_path = args.confusion_matrix
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

    # Set up the test dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    kitti_test_set = KittiTestDataset(dir=os.path.join(input_dir, 'test'), transform=transform)
    test_loader = DataLoader(kitti_test_set, batch_size=batch_size, shuffle=False)

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Modify based on your binary classification task
    model.load_state_dict(torch.load(model_name))

    model.eval()
    model.to(device=device)

    test_yoda_classifier(model=model, test_loader=test_loader, confusion_matrix_path=confusion_matrix_path,
                         accuracy_path=accuracy_path, device=device)

    print("testing completed")
    exit(0)
