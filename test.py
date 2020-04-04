import gym
import torch
import random
import torchvision.transforms as T
import numpy as np

from collections import namedtuple

Experience = namedtuple('Experience',
                    ('state', 'action', 'reward', 'next_state'))

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
    
    def num_actions(self):
        return self.env.action_space.n

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

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.count = 0
    
    def add_to_memory(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else: 
            self.memory[self.capacity % self.count] = experience
        self.count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


if __name__ == "__main__":
    envmanager = AtariBreakoutEnvManager(device='cpu')
    memory = ReplayMemory(capacity=1000000)

    nb_games = 50

    for i in range(nb_games):
        envmanager.reset()
        
        for i in range(256):
            # env.render(mode="human")
            action = envmanager.env.action_space.sample()
            observation, reward, done, info = envmanager.env.step(action)
            observation = envmanager.get_process()
            memory.add_to_memory(Experience(observation, action, reward, observation))

            if memory.can_provide_sample(256):
                experiences = memory.sample(256)
            
    import ipdb; ipdb.set_trace()
    env.close()