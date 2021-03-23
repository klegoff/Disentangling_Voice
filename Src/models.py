"""
class for Neural network models
"""

import torch.nn as nn

import torch.nn.functional as F

class spectrogram_model(nn.Module):
    def __init__(self, n_out):
        """
        n_out = number of values for the chosen variable (ex : for age, n_out=3)
        """
        super(spectrogram_model, self).__init__()
        # conv layers
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)

        # max pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # activation
        self.act1 = nn.Linear(32 * 19 * 6, 120)
        self.act2 = nn.Linear(120, 60)
        self.act3 = nn.Linear(60, n_out)
        
        # dropout
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # shape = 10, 8, 90, 39
        x = self.drop(x)
        x = self.pool(F.relu(self.conv2(x))) # shape = 10, 16, 43, 17
        x = self.drop(x)
        x = self.pool(F.relu(self.conv3(x))) # shape = 10, 32, 19, 6
        x = self.drop(x)
        x = x.view(-1, 32 * 19 * 6)
        x = F.relu(self.act1(x))
        x = F.relu(self.act2(x))
        x = self.act3(x)
        return x
