import numpy as np
import gym
import torch
import torchvision.transforms as T

from collections import deque

class ProcessFrame84(gym.ObservationWrapper):

    def __init__(self, device, env):
        super(ProcessFrame84, self).__init__(env)

        self.device = device
        self.env = env
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

    def get_transform(self, obs):
        obs = np.ascontiguousarray(obs, dtype=np.float32) / 255
        obs = torch.from_numpy(obs)

        convert = T.Compose([T.ToPILImage(),
                             T.Grayscale(),
                             T.Resize((110, 84)),
                             T.ToTensor()])

        return convert(obs)

    def get_crop(self, obs):
        top = 17
        bottom = 101
        obs = obs[:, top:bottom, :]

        return obs

    def observation(self, obs):
        obs = obs.transpose((2, 0, 1))
        # Reshape to (110, 84) and RGB to GrayScale
        obs = self.get_transform(obs)
        # Crop top and bottom to obtain shape (84, 84)
        obs = self.get_crop(obs)

        return obs.permute((1, 2, 0)) # obs.shape is (84, 84, 1)

    def get_height(self):
        obs = self.get_state()
        return obs.shape[2]

    def get_width(self):
        obs = self.get_state()
        return obs.shape[3]


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
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

if __name__ == "__main__":

    env = gym.make("Breakout-v0")
    env = ProcessFrame84("cpu", env)
    env = MaxAndSkipEnv(env)
    env = BufferWrapper(env, 4)
    obs = env.reset()

    env.observation(obs)