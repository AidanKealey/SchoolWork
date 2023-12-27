import torch
import os
import cv2
import argparse
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
from KittiDataset import KittiDataset
from KittiAnchors import Anchors
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

save_ROIs = True
max_ROIs = -1
ROI_IoUs = []

def strip_ROIs(class_ID, label_list):
    ROIs = []

    # Convert label_list to a list if it's a tensor
    if isinstance(label_list, torch.Tensor):
        label_list = label_list.item() if label_list.numel() == 1 else label_list.tolist()

    # Ensure label_list is iterable
    if not isinstance(label_list, (list, tuple)):
        label_list = [label_list]  # Convert to a list if it's an integer

    for i in range(len(label_list)):
        ROI = label_list[i]

        # Check if ROI is a float
        if not isinstance(ROI, (float, int)):
            if ROI[1] == class_ID:
                pt1 = (int(ROI[3]), int(ROI[2]))
                pt2 = (int(ROI[5]), int(ROI[4]))
                ROIs += [(pt1, pt2)]

    return ROIs

def calc_IoU(boxA, boxB):
    # print('break 209: ', boxA, boxB)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0][1], boxB[0][1])
    yA = max(boxA[0][0], boxB[0][0])
    xB = min(boxA[1][1], boxB[1][1])
    yB = min(boxA[1][0], boxB[1][0])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[1][1] - boxA[0][1] + 1) * (boxA[1][0] - boxA[0][0] + 1)
    boxBArea = (boxB[1][1] - boxB[0][1] + 1) * (boxB[1][0] - boxB[0][0] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def calc_max_IoU(ROI, ROI_list):
    max_IoU = 0
    for i in range(len(ROI_list)):
        max_IoU = max(max_IoU, calc_IoU(ROI, ROI_list[i]))
    return max_IoU


def step4_1():
    print('running KittiToYoda ...')

    label_file = 'labels.txt'

    argParser = argparse.ArgumentParser()
    argParser.add_argument('-i', metavar='input_dir', type=str, help='input dir (./)')
    argParser.add_argument('-o', metavar='output_dir', type=str, help='output dir (./)')
    argParser.add_argument('-IoU', metavar='IoU_threshold', type=float, help='[0.02]')
    argParser.add_argument('-d', metavar='display', type=str, help='[y/N]')
    argParser.add_argument('-m', metavar='mode', type=str, help='[train/test]')
    argParser.add_argument('-cuda', metavar='cuda', type=str, help='[y/N]')

    args = argParser.parse_args()

    input_dir = None
    if args.i != None:
        input_dir = args.i

    output_dir = None
    if args.o != None:
        output_dir = args.o

    IoU_threshold = 0.02
    if args.IoU != None:
        IoU_threshold = float(args.IoU)

    show_images = False
    if args.d != None:
        if args.d == 'y' or args.d == 'Y':
            show_images = True

    training = True
    if args.m == 'test':
        training = False

    use_cuda = False
    if args.cuda != None:
        if args.cuda == 'y' or args.cuda == 'Y':
            use_cuda = True

    labels = []

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

    if use_cuda == True and torch.cuda.is_available():
        device = 'cuda'
    print('using device ', device)

    dataset = KittiDataset(input_dir, training=training)
    anchors = Anchors()

    i = 0
    for item in enumerate(dataset):
        idx = item[0]
        image = item[1][0]
        label = item[1][1]
        # print(i, idx, label)

        idx = dataset.class_label['Car']
        car_ROIs = strip_ROIs(class_ID=1, label_list=label[0])
        # print(car_ROIs)
        # for idx in range(len(car_ROIs)):
        # print(ROIs[idx])

        anchor_centers = anchors.calc_anchor_centers(image.shape, anchors.grid)
        if show_images:
            image1 = image.copy()
            for j in range(len(anchor_centers)):
                x = anchor_centers[j][1]
                y = anchor_centers[j][0]
                cv2.circle(image1, (x, y), radius=4, color=(255, 0, 255))


        ROIs, boxes = anchors.get_anchor_ROIs(image, anchor_centers, anchors.shapes)
        # print('break 555: ', boxes)

        ROI_IoUs = []
        for idx in range(len(ROIs)):
            ROI_IoUs += [anchors.calc_max_IoU(boxes[idx], car_ROIs)]

        # print(ROI_IoUs)

        for k in range(len(boxes)):
            filename = str(i) + '_' + str(k) + '.png'
            if save_ROIs == True:
                cv2.imwrite(os.path.join(output_dir, filename), ROIs[k])
            name_class = 0
            name = 'NoCar'
            if ROI_IoUs[k] >= IoU_threshold:
                name_class = 1
                name = 'Car'

            min_x = boxes[k][0][0]
            min_y = boxes[k][0][1]
            max_x = boxes[k][1][0]
            max_y = boxes[k][1][1]

            labels += [[filename, name_class, name, min_x, min_y, max_x, max_y]]

        if show_images:
            cv2.imshow('image', image1)

        if show_images:
            image2 = image1.copy()

            for k in range(len(boxes)):
                if ROI_IoUs[k] > IoU_threshold:
                    box = boxes[k]
                    pt1 = (box[0][1], box[0][0])
                    pt2 = (box[1][1], box[1][0])
                    cv2.rectangle(image2, pt1, pt2, color=(0, 255, 255))

        if show_images:
            cv2.imshow('boxes', image2)
            key = cv2.waitKey(0)
            if key == ord('x'):
                break

        i += 1
        print(i)

        if max_ROIs > 0 and i >= max_ROIs:
            break

    if save_ROIs == True:
        with open(os.path.join(output_dir, label_file), 'w') as f:
            for k in range(len(labels)):
                filename = labels[k][0]
                name_class = str(labels[k][1])
                name = labels[k][2]
                min_x = str(labels[k][3])
                min_y = str(labels[k][4])
                max_x = str(labels[k][5])
                max_y = str(labels[k][6])
                f.write(
                    filename + ' ' + name_class + ' ' + name + ' ' + min_x + ' ' + min_y + ' ' + max_x + ' ' + max_y + '\n')
        f.close()


###################################################################

class KittiTestDataset_step4_2(Dataset):
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

        if isinstance(img_path, torch.Tensor):
            image = transforms.ToPILImage()(img_path).convert('RGB')
        else:
            image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def run_model(model, test_loader, device):
    prediction_label = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            outputs = model(imgs)

            _, predictions = torch.max(outputs, 1)
            prediction_label.extend(predictions.cpu().numpy())

    return prediction_label


def image_groups(dataset):
    items = len(dataset)
    images = items // 48
    image_groups = [list(range(i * 48, (i + 1) * 48)) for i in range(images)]
    return image_groups


def step4_2n3(num_of_images, device):
    for nu in range(num_of_images):
    # calculated_IOUs_for_all_images = []
    # for nu in range(1480):
        random_image = random.randint(0, 1480)
        # random_image = nu
        print(f"Image {random_image} selected")

        # to get predicted labels
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # random_image = 0                # remove after testing

        # get original info
        dataset = KittiDataset(dir='./data/Kitti8', training=False)
        anchors = Anchors()
        show_images = True
        true_boxes = None

        for item in enumerate(dataset):
            x = item[0]
            if x == random_image:
                image = item[1][0]
                label = item[1][1]

                idx = dataset.class_label['Car']
                car_ROIs = strip_ROIs(class_ID=1, label_list=label[0])

                anchor_centers = anchors.calc_anchor_centers(image.shape, anchors.grid)

                true_ROIs, true_boxes = anchors.get_anchor_ROIs(image, anchor_centers, anchors.shapes)

        # get predicted labels
        kitti_test_set = KittiTestDataset_step4_2(dir='data/step4', transform=transform)

        images_kitti_test_set = image_groups(kitti_test_set)
        selected_image = images_kitti_test_set[random_image]
        selected_images_group = [kitti_test_set[idx] for idx in selected_image]
        selected_images, selected_labels = zip(*selected_images_group)

        selected_kitti_test_set = KittiTestDataset_step4_2(dir='data/step4', transform=transform)
        selected_kitti_test_set.img_files = selected_images
        selected_kitti_test_set.labels = selected_labels

        test_loader = DataLoader(selected_kitti_test_set, batch_size=48, shuffle=False)

        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load('yoda_classifier_train_e30.pth'))
        model.eval()
        model.to(device=device)

        prediction_box = []
        calculated_IOUs = []

        predictions = run_model(model=model, test_loader=test_loader, device=device)

        labels_for_image = []
        with open("data/step4/labels.txt", "r") as file:
            for line in file:
                if line.startswith(f'{random_image}_'):
                    labels_for_image.append(line)

        for i in range(len(predictions)):
            if predictions[i] == 1:
                label = labels_for_image[i].split()

                pred_min_x = int(label[3])
                pred_min_y = int(label[4])
                pred_max_x = int(label[5])
                pred_max_y = int(label[6])

                prediction_box.append([(pred_min_y, pred_min_x), (pred_max_y, pred_max_x)])

        for z in range(len(prediction_box)):
            calculated_IOUs.append(calc_max_IoU(prediction_box[z], true_boxes))

        print(np.mean(calculated_IOUs))
        # calculated_IOUs_for_all_images.append(np.mean(calculated_IOUs))

        if show_images:
            image1 = image.copy()

            for item in enumerate(dataset):
                x = item[0]
                if x == random_image:
                    image = item[1][0]
                    label = item[1][1]
                    for j in range(len(label)):
                        name = label[j][0]
                        name_class = label[j][1]
                        if name_class == 2:
                            minx = int(label[j][2])
                            miny = int(label[j][3])
                            maxx = int(label[j][4])
                            maxy = int(label[j][5])
                            cv2.rectangle(image1, (minx, miny), (maxx, maxy), (0, 0, 255))

            # for k in range(len(true_boxes)):
            #     box = true_boxes[k]
            #     pt1 = (box[0][1], box[0][0])
            #     pt2 = (box[1][1], box[1][0])
            #     cv2.rectangle(image1, pt1, pt2, color=(0, 255, 255))

            image2 = image1.copy()

            for k in range(len(prediction_box)):
                box = prediction_box[k]
                # pt1 = (box[0][1], box[0][0])
                # pt2 = (box[1][1], box[1][0])
                pt1 = (box[0][0], box[0][1])
                pt2 = (box[1][0], box[1][1])
                cv2.rectangle(image2, pt1, pt2, color=(255, 165, 0))

        if show_images:
            cv2.imshow('boxes', image2)
            key = cv2.waitKey(0)
            if key == ord('x'):
                cv2.destroyAllWindows()

        nu += 1

    # print(np.mean(calculated_IOUs_for_all_images))

###################################################################

if __name__ == "__main__":
    print("Setting up PyTorch and device...")
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

    # step4_1() ## uncomment and run to create dataset, also look at function for parameters. It should be similar to
    # KittiToYodaROIs.py
    step4_2n3(num_of_images=2, device=device)

    exit(0)