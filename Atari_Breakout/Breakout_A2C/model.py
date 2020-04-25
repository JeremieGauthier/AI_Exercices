import numpy as np 
import torch.nn as nn
import torch

class A2C(nn.Module):
    def __init__(self, input_size, num_actions):
        super(A2C, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_size[0], 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU()
        )

        self.conv_output_size = self.get_output_size(input_size)

        self.actor = nn.Sequential(
            nn.Linear(self.conv_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(self.conv_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
    
    def get_output_size(self, input_size):
        output = self.conv(torch.zeros(1, *input_size))
        return int(np.prod(*output.shape))

    def forward(self, state):
        state = state.float() / 256
        conv_out = self.conv(state).reshape(-1, self.conv_output_size)
        
        policy_net = self.actor(conv_out)
        value_net = self.critic(conv_out)

        return policy_net, value_net