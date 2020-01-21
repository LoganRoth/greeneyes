import torch.nn as nn
import torch.nn.functional as F


# define the CNN architecture
# 94% - 100%
class BestSoFarNet(nn.Module):
    def __init__(self):
        super(BestSoFarNet, self).__init__()
        # convolutional layer (sees 266x266x3 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 11)  # -> 256x256x16
        # convolutional layer (sees 64x64x64 tensor)
        self.conv2 = nn.Conv2d(16, 64, 3, padding=1)  # -> 64x64x64
        # convolutional layer (sees 16x16x64 tensor)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # -> 16x16x128
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # max pooling layer
        self.pool4 = nn.MaxPool2d(4, 4)
        # linear layer (128 * 8 * 8 -> 500)
        self.fc1 = nn.Linear(128 * 8 * 8, 500)
        # linear layer (500 -> 6)
        self.fc2 = nn.Linear(500, 7)
        # dropout layer (p=0.15)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool4(F.relu(self.conv1(x)))
        x = self.pool4(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(-1, 128 * 8 * 8)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu and add dropout layer
        x = self.dropout(F.relu(self.fc1(x)))
        # add 2nd hidden layer, with relu
        x = self.fc2(x)
        return x


# define the CNN architecture
# 89% - 100%
class OtherBestNet(nn.Module):
    def __init__(self):
        super(OtherBestNet, self).__init__()
        # convolutional layer (sees 266x266x3 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 11)  # -> 256x256x16
        # convolutional layer (sees 256x256x16 tensor)
        self.conv2 = nn.Conv2d(16, 64, 3, padding=1)  # -> 256x256x64
        # convolutional layer (sees 64x64x64 tensor)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # -> 64x64x128
        # convolutional layer (sees 16x16x128 tensor)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)  # -> 16x16x256
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # max pooling layer
        self.pool4 = nn.MaxPool2d(4, 4)
        # linear layer (256 * 8 * 8 -> 800)
        self.fc1 = nn.Linear(256 * 8 * 8, 800)
        # linear layer (800 -> 6)
        self.fc2 = nn.Linear(800, 7)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = F.relu(self.conv1(x))
        x = self.pool4(F.relu(self.conv2(x)))
        x = self.pool4(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        # flatten image input
        x = x.view(-1, 256 * 8 * 8)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu and add dropout layer
        x = self.dropout(F.relu(self.fc1(x)))
        # add 2nd hidden layer, with relu
        x = self.fc2(x)
        return x
