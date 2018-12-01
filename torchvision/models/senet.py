import torch
import torch.nn as nn
import torch.nn.functional as F
'''
https://github.com/Marcovaldong/LightModels.git
'''
class BasicBlock(nn.Module):
    def __init__(self, in_channels, channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels))

        # SE layers
        self.fc1 = nn.Conv2d(channels, channels//16, kernel_size=1) # use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(channels//16, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = self.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += self.shortcut(x)
        out = self.relu(out)
        return out


class PreActBlock(nn.Module):
    def __init__(self, in_channels, channels, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride, bias=False))

        # SE layer
        self.fc1 = nn.Conv2d(channels, channels//16, kernel_size=1)
        self.fc2 = nn.Conv2d(channels//16, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(self.relu(self.bn2(out)))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = self.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut
        return out

class SENet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(SENet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        self.linear = nn.Linear(512, num_classes)
        self.relu = nn.ReLU(True)

    def _make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def name(self):
        return 'SENet'

def SENet18( **kwargs):
    model = SENet(PreActBlock, [2, 2, 2, 2],**kwargs)
    return model
