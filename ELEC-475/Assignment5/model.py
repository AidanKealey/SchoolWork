import torch.nn as nn
from torchvision import models


class NoseRegressor(nn.Module):
    def __init__(self):
        super(NoseRegressor, self).__init__()
        resnet18 = models.resnet18(weights=True)
        self.features = nn.Sequential(*list(resnet18.children())[:-1])
        self.regressor = nn.Linear(512, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)

        return x

