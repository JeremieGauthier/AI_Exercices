import wrappers
import utils
import model

import gym
import time
import math
import ptan
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from itertools import count
from torch.utils.tensorboard import SummaryWriter


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
            "max_nb_elements" : 4,
        },
    }

    params = HYPERPARAMS["breakout"]

    scores, eps_history = [], []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = wrappers.make_env(params["env_name"])

    policy_network = model.NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)
    target_network = ptan.agent.TargetNet(policy_network)
    optimizer = optim.Adam(policy_network.parameters(), lr=params["learning_rate"])

    action_selector = ptan.actions.ArgmaxActionSelector()
    agent = ptan.agent.DQNAgent(policy_network, action_selector, device)

    exp_source  = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params["gamma"], steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params["capacity"])

    writer = SummaryWriter("run")

    current_step = 0

    with utils.RewardTracker(writer, params) as reward_tracker:
        for episode in count():

            buffer.populate(1)
            current_step += 1

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], current_step):
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

                target_q_value = rewards + params["gamma"] * next_q_value.detach()

                loss = nn.MSELoss()
                loss = loss(target_q_value, current_q_value)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if episode % params["target_update"] == 0:
                target_network.sync()
            
