import os
import argparse
import torch
import time
import model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms
from torchvision.transforms import functional
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class nose_dataset(Dataset):
    def __init__(self, dir_txt, dir_imgs, transform=None):
        self.dir_txt = dir_txt
        self.dir_imgs = dir_imgs
        self.points = pd.read_csv(self.dir_txt)
        self.transform = transform

    def __len__(self):
        return len(self.points)

    def __getitem__(self, item):
        img_path = os.path.join(self.dir_imgs, self.points.iloc[item, 0])
        img_nose = eval(self.points.iloc[item, 1])

        img = Image.open(img_path).convert('RGB')

        x, y = map(int, img_nose)
        img_size = img.size
        new_x = int(x * (256 / img_size[0]))
        new_y = int(y * (256 / img_size[1]))
        norm_x = (new_x / 256) * 2 - 1
        norm_y = (new_y / 256) * 2 - 1

        nose_coordinates = torch.tensor(data=[norm_x, norm_y], dtype=torch.float32)

        if self.transform:
            data = {"image": img, "points": img_nose}
            data = self.transform(data)
            img, img_nose = data["image"], data["points"]

        return img_path, img, nose_coordinates


class resize_with_points(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image = data["image"]
        points = data["points"]

        resize_image = functional.resize(image, self.size)
        resize_image = functional.to_tensor(resize_image)

        resize_points = [p * self.size[i] / image.size[i] for i, p in enumerate(points)]

        return {"image": resize_image, "points": resize_points}


def test(model, test_loader, device):
    print("testing... ")

    inference_times = []
    localization_errors = []

    start_time = time.time()
    with torch.no_grad():
        i = 0
        for _, (img_path, imgs, points) in enumerate(test_loader):
            batch_start_time = time.time()
            imgs = imgs.to(device=device)
            points = points.to(device=device)

            outputs = model(imgs)

            pred_points = outputs.detach().cpu().numpy()
            actual_coords = points.detach().cpu().numpy()

            for i in range(len(outputs)):
                img_name = img_path[i]
                predicted_nose = pred_points[i]
                noses = actual_coords[i]
                img = Image.open(img_name)
                img_size = img.size

                test_noses = 'data/test_noses.txt'
                image = os.path.basename(img_name)

                with open(test_noses, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        if line.split(',')[0] == image:
                            actual_nose = eval(line.split('"')[1])

                # Unnormalize the noses
                predicted_x, predicted_y = predicted_nose
                predicted_x = ((predicted_x + 1) / 2) * 256
                predicted_y = ((predicted_y + 1) / 2) * 256

                predicted_x = int(predicted_x * (img_size[0] / 256))
                predicted_y = int(predicted_y * (img_size[1] / 256))

                actual_x, actual_y = actual_nose

                distance = np.linalg.norm([predicted_x, predicted_y] - np.array(actual_nose))
                localization_errors.append(distance)

                plt.imshow(img)
                plt.scatter([actual_x], [actual_y], c='red', marker='X', label='Actual Coordinates')
                plt.scatter([predicted_x], [predicted_y], c='green', marker='X', label='Predicted Coordinates')
                plt.legend(loc='upper right', fontsize=8)
                plt.title('Image with Coordinates')

                plt.savefig(f'outputs/testing/images/predicted_{image}.png')
                # plt.show()
                plt.close()

            distance = np.linalg.norm(pred_points - actual_coords)
            localization_errors.append(distance)

            inference_time = (time.time() - batch_start_time)
            inference_times.append(inference_time)


    end_time = time.time()
    total_testing_time = end_time - start_time
    print(f'Testing Time: {total_testing_time} seconds -- about {int(total_testing_time / 60)} minutes')

    # Calculate localization accuracy statistics
    min_distance = np.min(localization_errors)
    mean_distance = np.mean(localization_errors)
    max_distance = np.max(localization_errors)
    std_distance = np.std(localization_errors)

    # Calculate inference time statistics
    min_inference_time = np.min(inference_times)
    mean_inference_time = np.mean(inference_times)
    max_inference_time = np.max(inference_times)
    std_inference_time = np.std(inference_times)

    print(f"Localization Accuracy Statistics:")
    print(f"Min Distance: {min_distance}")
    print(f"Mean Distance: {mean_distance}")
    print(f"Max Distance: {max_distance}")
    print(f"Standard Deviation: {std_distance}")

    print("\nInference Time Statistics:")
    print(f"Min Inference (calculation + Image/nose printing) Time: {min_inference_time:.2f} s per 48 image")
    print(f"Mean Inference (calculation + Image/nose printing) Time: {mean_inference_time:.2f} s per 48 image")
    print(f"Max Inference (calculation + Image/nose printing) Time: {max_inference_time:.2f} s per 48 image")
    print(f"Standard Deviation: {std_inference_time:.2f} s per 48 image")


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-t', '--input_dir_txt', type=str, help='input dir (./)', default='./')
    argParser.add_argument('-i', '--input_dir_img', type=str, help='input dir (./)', default='./')
    argParser.add_argument('-m', '--model', type=str, default='yoda_classifier_trained.pth', help='model name')
    argParser.add_argument('-b', '--batch_size', type=int, default=48, help='batch size')

    args = argParser.parse_args()

    input_dir_txt = args.input_dir_txt
    input_dir_img = args.input_dir_img
    batch_size = args.batch_size                # 48
    model_name = args.model                     # 'nose_regress_e12.pth'

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
    print(f"Using device: {device}")

    transform = transforms.Compose([
        resize_with_points((256, 256))
    ])

    testing_set = nose_dataset(dir_txt=input_dir_txt, dir_imgs=input_dir_img, transform=transform)

    test_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=True)

    model = model.NoseRegressor()
    model.load_state_dict(torch.load(model_name))

    model.eval()
    model.to(device=device)

    test(model=model, test_loader=test_loader, device=device)

    print("testing completed")
    exit(0)
