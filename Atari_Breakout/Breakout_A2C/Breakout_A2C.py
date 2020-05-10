from model import A2C
from agent import Agent
from wrappers import make_env
from experience import ExperienceSourceFirstLast, unpack_batch
from common import RewardTracker

import numpy as np 
import torch.optim as optim
import torch as T
import gym

from itertools import count
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":

    HYPERPARAMS = {
        "breakout": {
            "env_name": "BreakoutNoFrameskip-v4",
            "gamma": 0.99, 
            "learning_rate": 0.0001,
            "entropy_beta": 0.001,
            "batch_size": 8,
            "accumulation_steps": 10,
            "reward_steps": 4,
            "stop_reward": 500,
        }
    }

    params = HYPERPARAMS["breakout"]

    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    env = make_env(params["env_name"])
    writer = SummaryWriter("run")

    net = A2C(env.observation_space.shape, env.action_space.n)
    optimizer = optim.Adam(net.parameters(), lr=params["learning_rate"])

    agent = Agent(net, params["batch_size"], params["entropy_beta"], params["accumulation_steps"])
    exp_source = ExperienceSourceFirstLast(env, agent, params["gamma"], params["reward_steps"])
    
    batch = []

    with RewardTracker(writer, stop_reward=params["stop_reward"]) as tracker:
        for step, exp in enumerate(exp_source):
            batch.append(exp)

            #This part is only used to track the total reward.
            #If new_reward=True, it means the episode is done
            new_reward = exp_source.pop_total_reward()
            if new_reward:
                if tracker.reward(new_reward[0], step):
                    break
            
            if len(batch) < params["batch_size"]:
                continue 

            # Output the tuple (batch_states, batch_actions, batch_qvals)
            # batch_args = unpack_batch(batch, net, params["gamma"], params["reward_steps"], device=device)
            batch_states_ts, batch_actions_ts, batch_qvals_ts = unpack_batch(batch, net, params["gamma"], params["reward_steps"], device=device)
            batch.clear()

            # agent.learn(step, *batch_args, optimizer)
            agent.learn(step, batch_states_ts, batch_actions_ts, batch_qvals_ts, optimizer)