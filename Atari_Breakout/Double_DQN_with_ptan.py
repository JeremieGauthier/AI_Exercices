from model import DQN

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

"""
In basic DQN, the optimal polcy of the agent is always to choose the best
action in any given state. The assumption behind the idea is that the best
action has the best expected Q-value. However, the agent knows nothing 
about the environment at the beginning. It needs to estimate Q(s,a) and 
update them at each iteration. Such Q-values have lot of noises and we are
never sure if the action with maximum expected Q-value is really the best one
"""

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
            "env_name" : "BreakoutNoFrameskip-v4",
            "learning_rate" : 0.0001,
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
            "reward_steps" : 1,
        },
    }

    params = HYPERPARAMS["breakout"]

    scores, eps_history = [], []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = wrappers.make_env(params["env_name"])

    policy_network = DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_network = ptan.agent.TargetNet(policy_network)
    optimizer = optim.Adam(policy_network.parameters(), lr=params["learning_rate"])

    action_selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params["eps_start"])
    agent = ptan.agent.DQNAgent(policy_network, action_selector, device)
    epsilon_tracker = EpsilonTracker(action_selector, params)

    exp_source  = ptan.experience.ExperienceSourceFirstLast(env, agent,
                  gamma=params["gamma"], steps_count=params["reward_steps"])
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params["capacity"]) 
    writer = SummaryWriter("run")

    current_step = 0

    with utils.RewardTracker(writer, params) as reward_tracker:
        for episode in count():

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

                ############################ Double DQN part #############################
                next_state_actions = policy_network(next_states).max(1)[1]
                next_q_value = target_network.target_model(next_states).gather(1, 
                            next_state_actions.unsqueeze(-1)).squeeze(-1)

                # Normally, only the line below is used for that task
                # next_q_value = target_network.target_model(next_states).max(1)[0]
                ########################################################################## 

                next_q_value[done_mask] = 0.0

                target_q_value = rewards + params["gamma"] \
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
