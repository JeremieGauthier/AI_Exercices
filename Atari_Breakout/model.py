import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

        import ipdb; ipdb.set_trace()
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
        
        return self.linear(self.layer1)



class NoisyLinear(nn.Linear): #Independent Gaussian Noise used with NoisyDQN model
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)

        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))

        if bias: 
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3/self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        weight = self.weight + self.sigma_weight * self.epsilon_weight
        if self.bias is not None:
            self.epsilon_bias.normal_()
            bias = self.bias + self.sigma_bias * self.epsilon_bias
        
        return F.linear(input, weight, bias)


class NoisyFactorizedLinear(nn.Linear): #Factorized Gaussian Noise used with NoisyDQN model
    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)

        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.full(out_features, in_features), sigma_init)

        self.register_buffer("epsilon_input", torch.zeros((1, in_features)))
        self.register_buffer("epsilon_output", torch.zeros((out_features, 1)))

        if bias:
            self.sigma_bias = nn.Parameter(torch.full(out_features,), sigma_init)

    def forward(self, input):
        pass


class NoisyDQN(nn.Module):
    """
    Look at https://arxiv.org/pdf/1706.10295v3.pdf
    """

    def __init__(self, input_shape, num_actions):
        super(NoisyDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.conv_output_size = self.get_output_size(input_shape)

        self.linear = nn.Sequential(
           NoisyLinear(in_features=self.conv_output_size, out_features=512),
           nn.ReLU(),
           NoisyLinear(in_features=512, out_features=num_actions)
        )

    def get_output_size(self, input_shape):
        output = self.conv(torch.zeros((1, *input_shape)))
        return int(np.prod(output.shape))

    def forward(self, input):
        self.layer1 = self.conv(input)
        self.layer1 = self.layer1.reshape(-1, self.conv_output_size)

        return self.linear(self.layer1)
