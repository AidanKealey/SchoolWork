import torchvision.models as models
from torch.optim import lr_scheduler
# from prodigyopt import Prodigy
from torch.utils.data import DataLoader
import functions
import torchvision.transforms as transform
import torch

batch_size, n_epochs, weight_file, loss_image, device, weight_decay = functions.parse_args()

# Data augmentation
train_transform = transform.Compose([transform.RandomCrop(32, padding=4),
                                     transform.RandomHorizontalFlip(),
                                     transform.ToTensor(),
                                     transform.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
test_transform = transform.Compose([transform.ToTensor()])

# Dataset
train_data = functions.custom_dataset('./data/Kitti8_ROIs/train/', './data/Kitti8_ROIs/train/labels.txt', transform=train_transform)
test_data = functions.custom_dataset('./data/Kitti8_ROIs/test/', './data/Kitti8_ROIs/test/labels.txt', transform=train_transform)

train_dl = DataLoader(train_data, batch_size, shuffle=True)
test_dl = DataLoader(test_data, batch_size, shuffle=False)

# Model instantiation
model = models.resnet18()
model.to(device)

# Optimizer and scheduler instantiation
optimizer = Prodigy(model.parameters(), lr=1., weight_decay=weight_decay, decouple=True, safeguard_warmup=True)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

functions.train(model, n_epochs, train_dl, test_dl, device, optimizer, scheduler, loss_fn=torch.nn.CrossEntropyLoss(), loss_file=loss_image)

