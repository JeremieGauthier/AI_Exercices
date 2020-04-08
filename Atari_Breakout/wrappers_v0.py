import gym
import time
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

from collections import namedtuple, deque
from itertools import count


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

    def get_reward(self, action):
        # Here action must be of type torch.tensor
        _, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward]).to(self.device)

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

    def get_state(self):
        img = self.env.render("rgb_array").transpose((2, 0, 1))
        # Reshape to (110, 84) and RGB to GrayScale
        img = self.get_transform(img)
        # Crop top and bottom to obtain shape (84, 84)
        img = self.get_crop(img)

        return img.unsqueeze(dim=0)

    def get_height(self):
        img = self.get_state()
        return img.shape[2]

    def get_width(self):
        img = self.get_state()
        return img.shape[3]


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

    
class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer