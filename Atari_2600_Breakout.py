import gym
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

from PIL import Image

class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=1)

class AtariBreakout():
    def __init__(self, device):
        self.device = device
        self.env = gym.make('Breakout-v0').unwrapped
        self.env.reset()
        self.current_screen = None
        self.done = False
    
    def reset(self):
        self.env.reset()
        self.current_screen = None
    
    def render(self, mode="human"):
        self.env.render(mode)

    def close(self):
        self.env.close()

    def get_transform(self, img):
        img = np.ascontiguousarray(img, dtype=np.float32) / 255

    def get_process(self):
        img = self.render("rgb_array").transpose((2, 0, 1))
        img = torch.from_numpy(img)

    def get_height(self):
        screen = self.get_processed()
        return img.shape[2]
    
    def get_width(self):
        img = self.get_processed_screen()
        return img.shape[3]



if __name__ == "__main__":

    env = gym.make('Breakout-v0')
    nb_games = 1000
    done = False
    
    for i in range(nb_games):
        env.reset()
        
        while not done:
            env.render(mode="human")
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            import ipdb; ipdb.set_trace()
    env.close()
