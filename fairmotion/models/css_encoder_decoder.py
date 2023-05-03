import torch
import torch.nn as nn


class cssEncoderShort(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(cssEncoderShort, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(2, 7), stride=2, padding=1)
        self.conv2 = nn.Conv2d(
            64, 128, kernel_size=(2, 7), stride=2, padding=1)
        self.conv3 = nn.Conv2d(
            128, 128, kernel_size=(2, 7), stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.linear_layer = nn.Linear(6144, hidden_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # Encoder
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.linear_layer(x)
        return x


class cssEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(cssEncoder, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(2, 7), stride=2, padding=1)
        self.conv2 = nn.Conv2d(
            64, 128, kernel_size=(2, 7), stride=2, padding=1)
        self.conv3 = nn.Conv2d(
            128, 128, kernel_size=(2, 7), stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.linear_layer = nn.Linear(49152, hidden_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # Encoder
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.linear_layer(x)
        return x


class cssDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(cssDecoder, self).__init__()
        # Decoder
        self.linear_layer1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear_layer2 = nn.Linear(hidden_dim, input_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # Decoder
        x = self.relu(self.linear_layer1(x))
        x = self.linear_layer2(x)
        return x
