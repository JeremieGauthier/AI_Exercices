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


class DQN(nn.Module):
    def __init__(self, num_actions, lr):
        super(DQN, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=4, stride=2)

        # You have to respect the formula ((W-K+2P/S)+1)
        self.fc = nn.Linear(in_features=32*9*9, out_features=256)
        self.out = nn.Linear(in_features=256, out_features=num_actions)


    def forward(self, state):
        # (1) Hidden Conv. Layer
        self.layer1 = F.relu(self.conv1(state))

        # (2) Hidden Conv. Layer
        self.layer2 = F.relu(self.conv2(self.layer1))
        
        # (3) Hidden Linear Layer
        input_layer3 = self.layer2.reshape(-1, 32*9*9)
        self.layer3 = self.fc(input_layer3)

        # (4) Output
        actions = self.out(self.layer3)

        return actions


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

    def extract_tensor(self, experiences):
        batch = Experience(*zip(*experiences))

        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        next_actions = torch.cat(batch.next_state)

        return (states, actions, rewards, next_actions)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


class EpsilonGreedyStrategy():
    def __init__(self, eps_start, eps_end, eps_decay):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def get_exploration_rate(self, current_step):
        return self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1 * current_step * self.eps_decay)


class Agent():
    def __init__(self, num_actions, strategy, device):
        self.strategy = strategy
        self.num_actions = num_actions
        self.current_step = 0
        self.device = device

    def choose_action(self, state, policy_net):
        self.epsilon = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if np.random.random() < self.epsilon:  # Explore
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)
        else:  # Exploit
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device)


class Phi(): #Store max_nb_elements consecutive observations
    def __init__(self, max_nb_elements):
        self.memory = deque()
        self.counter = 1
        self.max_nb_elements = max_nb_elements

    def add_to_memory(self, observation):
        if len(self.memory) >= self.max_nb_elements:
            diff = (len(self.memory) - self.max_nb_elements) + 1
            for i in range(diff):
                self.memory.popleft()
        self.memory.append(observation)

    def clear_memory(self):
        self.memory.clear()

            
if __name__ == "__main__":
    lr = 0.001
    gamma = 0.99
    eps_start = 1
    eps_end = 0.01
    eps_decay = 0.001
    target_update = 10
    num_episodes = 10000
    batch_size = 256
    capacity = 1000000
    max_nb_elements = 4
    scores= []

    Experience = namedtuple('Experience',
                            ('state', 'action', 'reward', 'next_state'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    envmanager = AtariBreakoutEnvManager(device)
    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
    agent = Agent(envmanager.num_actions(), strategy, device)
    memory = ReplayMemory(capacity)
    phi = Phi(max_nb_elements)

    policy_network = DQN(envmanager.num_actions(), lr).to(device)
    target_network = DQN(envmanager.num_actions(), lr).to(device)

    target_network.load_state_dict(policy_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(params=policy_network.parameters(), lr=lr)
    
    for episode in range(num_episodes):
        envmanager.reset()
        init_observation= envmanager.get_state()
        phi.clear_memory()
        phi.add_to_memory(init_observation)

        score = 0
        state = None

        for timestep in count():
            action = agent.choose_action(state, policy_network)
            reward = envmanager.get_reward(action)
            observation = envmanager.get_state()
            phi.add_to_memory(observation)

            if len(phi.memory) % 4 == 0 : # Every fourth screenshot is considered

                if state is None: 
                    state = torch.zeros([1, 4, 84, 84]) #Initial state
                next_state = torch.cat(tuple(phi.memory), dim=1) #Stack 4 consecutives observations

                memory.add_to_memory(Experience(state, action, reward, next_state))
                state = next_state
                
                score += reward
                scores.append(score)

                if memory.can_provide_sample(batch_size):
                    experiences = memory.sample(batch_size)
                    states, actions, rewards, next_states = memory.extract_tensor(experiences)

                    batch_index = np.arange(batch_size, dtype=np.int32)
                    current_q_value = policy_network.forward(states)[batch_index, actions.type(torch.LongTensor)]
                    next_q_value = target_network.forward(next_states)
                    target_q_value = rewards + gamma * torch.max(next_q_value, dim=1)[0]

                    loss = nn.MSELoss()
                    loss = loss(target_q_value, current_q_value).to(device)
                    optimizer.zero_grad()
                    loss.backward
                    optimizer.step()

        if episode % target_update == 0:
            target_network.load_state_dict(policy_network.state_dict())

        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print("episode", episode, "score %.1f average score %.1f epsilon %.2f" %
               (score, avg_score, agent.epsilon))

