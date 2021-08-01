import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 4) # in_channels, out_channels, kernel_size, stride
        # output = (x + 2*paddint - dilation * (kernel_size - 1)-1) / stride
        # output = (64, 28, 96)
        self.pool = nn.MaxPool2d(2, 2) # output = (64, 14, 48)
        self.drop1 = nn.Dropout()
        self.conv2 = nn.Conv2d(64, 16, 4) # output = (16, 10, 44) => (16, 5, 22)
        self.drop2 = nn.Dropout()

        self.fc1 = nn.Linear(16*5*22, 128)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop2(x)

        x = torch.Flatten(x, 1) # 배치를 제외하고 flatten() 하기 때문에 1
        x = F.relu(self.fc1(x))

        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, x):

        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()


    def forward(self, x):

        return x