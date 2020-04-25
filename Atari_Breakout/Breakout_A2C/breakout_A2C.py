import model

import numpy as np 
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import ptan
import gym

from torch.utils.tensorboard import SummaryWriter

def calc_qvals(rewards, gamma):
    sum_r = 0.0
    sum_rewards = []
    for r in reversed(rewards):
        sum_r *= gamma
        sum_r += r
    return reversed(sum_rewards)

if __name__ == "__main__":

    HYPERPARAMS = {
        "breakout": {
            "env_name": "BreakoutNoFrameskip-v4",
            "gamma": 0.99, 
            "learning_rate": 0.0001,
            "reward_steps": 10,
            "entropy_beta": 0.001,
            "baseline_step": 1000000
        }
    }

    params = HYPERPARAMS["breakout"]

    env = gym.make(params["env_name"])
    writer = SummaryWriter("run")

    actor_net, critic_net = model.A2C(env.observation_space.shape, env.action_space.n)

    agent = ptan.agent.ActorCriticAgent(actor_net, 
            preprocessor=ptan.agent.float32_preprocessor, apply_softmax=True)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, 
                                gamma=params["gamma"])

    optimizer = optim.Adam(actor_net.parameters(), lr=params["learning_rate"])

    for step, exp in emuratate(exp_source):


