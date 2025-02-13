import torch.nn as nn
import torch.nn.functional as F


class CNN1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3)  # 1 input channel, 8 output channels, 3x3 kernel
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Maxpooling with stride 2

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)  # 8 input channels, 16 output channels, 3x3 kernel
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Maxpooling with stride 2

        # Fully Connected Layer
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 54 * 54, 32)  # Flattened output size from 16 channels and 6x6 spatial dimensions
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)  # Output 4 classes

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        # gia na vroume to megethos pou xreiazomaste sto fc1
        # print(x.shape)
        x = self.flatten(x)
        x = self.relu_fc1(self.fc1(x))
        x = self.fc2(x)

        return x



class CNN2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Convolutional Layer 1: 32 filters, 3x3, ReLU activation
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        # Max Pooling Layer with step 4
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)

        # Convolutional Layer 2: 64 filters, 3x3, ReLU activation
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()

        # Max Pooling Layer with step 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer 3: 128 filters, 3x3, ReLU activation
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU()

        # Max Pooling Layer with step 2
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer 4: 256 filters, 3x3, ReLU activation
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu8 = nn.ReLU()
        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu9 = nn.ReLU()

        # Max Pooling Layer with step 2
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer 5: 512 filters, 3x3, ReLU activation
        self.conv10 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu10 = nn.ReLU()

        # Max Pooling Layer with step 2
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer: Flatten followed by FC with 1024 neurons and ReLU
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 3 * 3, 1024)  # Assuming input images are 224x224
        self.relu_fc1 = nn.ReLU()

        # Output layer with num_classes outputs
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Pass through convolutional layers
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool1(x)

        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool2(x)

        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.pool3(x)

        x = self.relu7(self.conv7(x))
        x = self.relu8(self.conv8(x))
        x = self.relu9(self.conv9(x))
        x = self.pool4(x)

        x = self.relu10(self.conv10(x))
        x = self.pool5(x)

        # Flatten the output of the convolutional layers
        x = self.flatten(x)

        # Fully connected layers
        x = self.relu_fc1(self.fc1(x))
        x = self.fc2(x)

        return x



class BasicBlock(nn.Module):
    def __init__(self, n_in, n_filters, stride=1):
        super().__init__()

        self.stride = stride

        # First convolutional layer
        self.conv1 = nn.Conv2d(n_in, n_filters, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.relu1 = nn.ReLU()

        # Second convolutional layer
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.relu2 = nn.ReLU()

        # Shortcut for dimension matching when stride = 2
        self.shortcut = nn.Sequential()
        if stride == 2 or n_in != n_filters:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=n_in, out_channels=n_filters, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(n_filters)
            )
        self.relu3 = nn.ReLU()

    def forward(self, x):
        # Save the input for the residual connection
        identity = x

        # Pass through the first convolution, batch norm, and ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        # Pass through the second convolution and batch norm
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        # If a shortcut exists (stride=2 or channel mismatch), apply it to the input
        if self.shortcut is not None:
            identity = self.shortcut(x)

        # Add the shortcut (residual connection)
        out = out + identity

        # Final ReLU activation
        out = self.relu3(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=4):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.pool1 = nn.AvgPool2d(kernel_size=4)
        self.linear = nn.Linear(512*7*7, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool1(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out