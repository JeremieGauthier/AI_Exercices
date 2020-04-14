import wrappers
import utils

import gym
import time
import math
import random
import ptan
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

from itertools import count
from torch.utils.tensorboard import SummaryWriter

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

class EpsilonTracker():
    def __init__(self, epsilon_greedy_selector, params):

        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.eps_start = params["eps_start"]
        self.eps_end = params["eps_end"]
        self.eps_frame = params["eps_frame"]
        self.frame(0)
    
    def frame(self, current_step):
        self.epsilon_greedy_selector.epsilon = \
            max(self.eps_end, self.eps_start - current_step / 
                self.eps_frame)

    
def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.bool), np.array(last_states, copy=False)


if __name__ == "__main__":

    HYPERPARAMS={
        "breakout":{
            "env_name" : "Breakout-v0",
            "learning_rate" : 0.001,
            "gamma" : 0.99,
            "stop_reward" : 500.0, 
            "eps_start" : 1,
            "eps_end" : 0.05,
            "eps_frame" : 10**5,
            "target_update" : 1000,
            "num_episodes" : 1500,
            "batch_size" : 32,
            "replay_initial" : 10000,
            "capacity" : 100000,
            "reward_steps" : 2,
        },
    }

    params = HYPERPARAMS["breakout"]

    scores, eps_history = [], []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = wrappers.make_env("Breakout-v0")

    policy_network = DQN(env.action_space.n, params["learning_rate"]).to(device)
    target_network = ptan.agent.TargetNet(policy_network)
    optimizer = optim.Adam(policy_network.parameters(), lr=params["learning_rate"])

    action_selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params["eps_start"])
    agent = ptan.agent.DQNAgent(policy_network, action_selector, device)
    epsilon_tracker = EpsilonTracker(action_selector, params)

    exp_source  = ptan.experience.ExperienceSourceFirstLast(env, agent,
                  gamma=params["gamma"]**, steps_count=params["reward_steps"])
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params["capacity"]) writer = SummaryWriter("run")

    current_step = 0

    with utils.RewardTracker(writer, params) as reward_tracker:
        for episode in count():

            print("Episode : ", episode)

            buffer.populate(1)
            epsilon_tracker.frame(current_step)
            current_step += 1

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], current_step, action_selector.epsilon):
                    break

            if len(buffer) >= params["batch_size"]:
                batch = buffer.sample(params["batch_size"])
                states, actions, rewards, dones, next_states = unpack_batch(batch)

                states = torch.tensor(states).to(device)
                next_states = torch.tensor(next_states).to(device)
                actions = torch.tensor(actions).to(device)
                rewards = torch.tensor(rewards).to(device)
                done_mask = torch.BoolTensor(dones).to(device)

                current_q_value = policy_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
                next_q_value = target_network.target_model(next_states).max(1)[0]
                next_q_value[done_mask] = 0.0

                target_q_value = rewards + params["gamma"]**params["reward_steps"] \
                    * next_q_value.detach()

                loss = nn.MSELoss()
                loss = loss(target_q_value, current_q_value)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if episode % params["target_update"] == 0:
                target_network.sync()
            
            #if episode != 0 and episode % 200 == 0:
                #save = {'state_dict': policy_network.state_dict(), 'optimizer': optimizer.state_dict()}
                #torch.save(save, "DQN_model_" + str(episode) + ".pkl")
