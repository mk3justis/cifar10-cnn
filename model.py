import torch
import torch.nn as nn

class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()
        # Initialize the layers
        self.convolution1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convolution2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten(start_dim=1)
        self.full_conn1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.full_conn2 = nn.Linear(in_features=120, out_features=80)
        self.full_conn3 = nn.Linear(in_features=80, out_features=10)
    def forward(self, x):
        # Define the dataflow through the layers
        x = self.maxpool1(self.relu(self.convolution1(x)))
        x = self.maxpool2(self.relu(self.convolution2(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu(self.full_conn1(x)))
        x = self.dropout(self.relu(self.full_conn2(x)))
        x = self.full_conn3(x)
        return x