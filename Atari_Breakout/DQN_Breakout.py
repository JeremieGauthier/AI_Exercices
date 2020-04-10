import wrappers
import utils

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
        self.layer3 = F.relu(self.fc(input_layer3))

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
        rewards = torch.tensor(batch.reward)
        #rewards = torch.cat(batch.reward)
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



if __name__ == "__main__":

    lr = 0.001
    gamma = 0.99
    eps_start = 1
    eps_end = 0.01
    eps_decay = 0.001
    target_update = 10
    num_episodes = 1000
    batch_size = 256
    capacity = 1000000
    max_nb_elements = 4

    scores, eps_history = [], []


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = wrappers.make_env("Breakout-v0")

    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)

    agent = Agent(env.action_space.n, strategy, device)
    memory = ReplayMemory(capacity)

    policy_network = DQN(env.action_space.n, lr).to(device)
    target_network = DQN(env.action_space.n, lr).to(device)

    target_network.load_state_dict(policy_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(params=policy_network.parameters(), lr=lr)
    
    for episode in range(num_episodes):
        obs = env.reset()
        state = env.observation(obs)
        state = torch.tensor(state).unsqueeze(dim=0)
        
        score = 0
        start = time.time()

        for timestep in count():
            action = agent.choose_action(state, policy_network)
            next_state, reward, done, _ = env.step(action) 
            next_state = torch.tensor(next_state).unsqueeze(dim=0)
            memory.add_to_memory(Experience(state, action, reward, next_state))
            state = next_state
            
            score += reward

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
                loss.backward()
                optimizer.step()

            if done:
                break

        scores.append(score)
        eps_history.append(agent.epsilon)

        if episode % target_update == 0:
            target_network.load_state_dict(policy_network.state_dict())
        
        if episode != 0 and episode % 200 == 0:
            save = {'state_dict': policy_network.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(save, "DQN_model_" + str(episode)+ "_" 
                       + str(int(score)) + ".pkl")

        print("episode :", episode, "epsilon :", agent.epsilon, "score", score,
                "time :", time.time()-start)

        if episode % 20 == 0:
            avg_score = np.mean(scores[-20:])
            print("episode", episode, "score %.1f average score %.1f epsilon %.2f" %
               (score, avg_score, agent.epsilon))
    
        if episode % 10 == 0 and episode != 0:

            filename = 'Atari_Breakout_DQN.png'
            x = [i for i in range(episode+1)]
            utils.plot_learning_curve(x, scores, eps_history, filename)
