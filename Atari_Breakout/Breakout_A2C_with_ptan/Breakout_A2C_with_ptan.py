import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import torch.nn as nn
import torch as T 
import numpy as np
import ptan
import gym

from torch.utils.tensorboard import SummaryWriter

class AtariA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariA2C, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_output_size = self. _get_conv_out(input_shape)

        self.policy = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
    
        self.value = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(T.zeros(1, *shape))
        return int(np.prod(o.shape))
    
    def forward(self, x):
        x = x.float() / 256
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)

    
def unpack_batch(batch, net, device='cpu'):

    states, actions, rewards = [], [], []
    not_done_idx = []
    last_states = []

    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)