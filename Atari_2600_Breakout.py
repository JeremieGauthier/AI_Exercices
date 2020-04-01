import gym
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np


class AtariBreakoutEnvManager():

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
        img = torch.from_numpy(img)

        convert = T.Compose([T.ToPILImage(),
                         T.Grayscale(),
                         T.Resize((110, 84)),
                         T.ToTensor()])  
        
        return convert(img)

    def get_crop(self, img):
        top = 17
        bottom = 101
        img = img[:, top:bottom, :]
        
        return img

    def get_process(self):
        img = self.env.render("rgb_array").transpose((2, 0, 1))
        img = self.get_transform(img) #Reshape to (110, 84) and RGB to GrayScale
        img = self.get_crop(img) # Crop top and bottom to obtain shape (84, 84)

        return img.unsqueeze(dim=0)

    def get_height(self):
        img = self.get_process()
        return img.shape[2] 

    def get_width(self):
        img = self.get_process()
        return img.shape[3] 

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=4, stride=2)

        #You have to respect the formula ((W-K+2P/S)+1)
        self.fc = nn.Linear(in_features=32*9*9, out_features=256)
        self.out = nn.Linear(in_features=256, out_features=4)

    def forward(self, state):

        # (1) Hidden Conv. Layer
        self.layer1 = F.relu(self.conv1(state))

        #(2) Hidden Conv. Layer
        self.layer2 = F.relu(self.conv1(self.layer1))

        #(3) Hidden Linear Layer
        self.layer3 = self.fc(self.layer2)

        #(4) Output
        actions = self.out(self.layer3)

        return actions



        
if __name__ == "__main__":

    
    nb_games = 1000
    done = False
    
    for i in range(nb_games):
        env = gym.make("Breakout-v0")
        env.reset()
        
        while not done:
            # env.render(mode="human")
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            import ipdb; ipdb.set_trace()
    env.close()
