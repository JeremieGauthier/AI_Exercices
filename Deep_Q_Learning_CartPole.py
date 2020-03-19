import gym
import math
import random
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import torchvision.transforms as T 

class DQN(nn.Module):

    def __init__(self, img_hight, img_width):
        super().__init__()

        self.fc1 = nn.Linear(in_features=img_hight*img_width*3, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=2)

    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(2))
        t = self.out(t)

        return t

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < len(self.capacity):
            self.memory.append(experience)
        else: 
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
            math.exp(-1. * current_step * self.decay)

class Agent():
    def __init__(self, strategy, num_actions, device):
        self. current_step  = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device

    def select_action(self, state, policy_net):
        rate = strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
           action = random.randrange(self.num_actions) #Explore
           return torch.tensor([action]).to(device)
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(device) #Exploit

class CartPoleEnvManager():
    def __init__(self, device):
        self.device = device
        self.env = gym.make('CartPole-v0').unwrapped
        self.env.reset()
        self.current_screen = None
        self.done = False

    def reset(self):
        self.env.reset()
        self.current_screen = None
    
    def close(self):
        self.env.close()

    def render(self, mode="humain"):
        return self.env.render(mode)

    def num_actions_available(self):
        