from model import A2C
from wrappers import make_env

import numpy as np 
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import ptan
import gym

from itertools import count
from collections import deque
from torch.utils.tensorboard import SummaryWriter

def calc_qvals(rewards, gamma):
    sum_r = 0.0
    sum_rewards = []
    for r in reversed(rewards):
        sum_r *= gamma
        sum_r += r
    return reversed(sum_rewards)

def memory(max_length):
    buffer = deque([], maxlen=max_lenth)


if __name__ == "__main__":

    HYPERPARAMS = {
        "breakout": {
            "env_name": "BreakoutNoFrameskip-v4",
            "gamma": 0.99, 
            "learning_rate": 0.0001,
            "reward_steps": 10,
            "entropy_beta": 0.001,
            "batch_size": 8,
            "baseline_step": 1000000
        }
    }

    params = HYPERPARAMS["breakout"]

    env = make_env(params["env_name"])
    writer = SummaryWriter("run")

    actor_net, critic_net = model.A2C(env.observation_space.shape, env.action_space.n)
    optimizer = optim.Adam(actor_net.parameters(), lr=params["learning_rate"])

    batch_states, batch_actions = [], []

    for episode in count():
        done = False
        score = 0





