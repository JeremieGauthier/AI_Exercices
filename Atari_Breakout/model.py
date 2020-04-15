import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1),
            nn.ReLU()
        )

        self.conv_output_size = self.get_output_size(input_shape)

        self.linear = nn.Sequential(
            nn.Linear(in_features=self.conv_output_size, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_actions)
        )
    
    def get_output_size(self, shape):
        output = self.conv(torch.zeros(1, *shape))
        return int(np.prod(output.shape))

    def forward(self, state):
        self.layer1 = self.conv(state)
        self.layer1 = self.layer1.reshape(-1, self.conv_output_size)
        
        actions = self.linear(self.layer1)

        return actions

