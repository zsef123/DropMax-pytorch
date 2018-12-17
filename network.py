import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, n_classes=10, last_feature=800):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 20, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 50, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(last_feature, n_classes)

    def forward(self, x):
        x = self.feature(x).view(x.shape[0], -1)
        return self.out(x)


class DropMaxCNN(CNN):
    def __init__(self, n_classes=10, last_feature=800):
        super(DropMaxCNN, self).__init__(n_classes, last_feature)

        self.o  = nn.Linear(last_feature, n_classes)
        self.ph = nn.Linear(last_feature, n_classes)
        self.rh = nn.Linear(last_feature, n_classes)

    def forward(self, x):
        x = self.feature(x).view(x.shape[0], -1)

        o = torch.sigmoid(self.o(x))
        ph, rh = self.ph(x), self.rh(x)

        p = torch.sigmoid(ph)
        r = torch.sigmoid(rh)
        # stop gradient on `ph`
        q = torch.sigmoid(ph.detach() + rh)
        return o, p, r, q

