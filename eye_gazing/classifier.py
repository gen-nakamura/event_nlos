import torch
import torch.nn as nn


class Classifier2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=4, concat=True, dropout=False, prob=0, down_size=2):
        super(Classifier2D, self).__init__()

        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels, channel_size, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(channel_size, channel_size, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(channel_size, 1, kernel_size=(3, 3), padding=(1, 1))
        
        # Define fully connected layers
        self.fc1 = nn.Linear(300, 32)#1849
        self.fc2 = nn.Linear(32, 2)
        
        # Define activation and pooling
        self.relu = nn.LeakyReLU()
        self.max_pool = nn.AvgPool2d(kernel_size=down_size)

        # Define dropout layers if dropout is True
        if dropout:
            self.dropout1 = nn.Dropout(p=prob)
            self.dropout2 = nn.Dropout(p=prob)
            self.dropout3 = nn.Dropout(p=prob)
        else:
            self.dropout1 = None
            self.dropout2 = None
            self.dropout3 = None

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        # Apply layers
        x = self._apply_layer(self.conv1, x, self.dropout1)
        # print('conv1 x shape: ',x.shape)
        x = self._apply_layer(self.conv2, x, self.dropout2)
        x = self._apply_layer(self.conv2, x, self.dropout2)
        x = self._apply_layer(self.conv2, x, self.dropout2)
        # print('conv2 x shape: ',x.shape)
        x = self._apply_layer(self.conv3, x, self.dropout3)
        # print('conv3 x shape: ',x.shape)
        # print('before flatten')
        x = torch.flatten(x, 1)
        # print('before relu')
        x = self.relu(self.fc1(x))
        # print('before return')
        return self.fc2(x)

    def _apply_layer(self, conv, x, dropout):
        """Apply convolution, dropout (if not None), activation and pooling."""
        x = conv(x)
        if dropout is not None:
            x = dropout(x)
        x = self.relu(x)
        return self.max_pool(x)

    def _initialize_weights(self):
        """Initialize weights for the modules."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.random_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
