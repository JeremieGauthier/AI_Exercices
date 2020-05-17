from model import A2C
from agent import Agent
from wrappers import GymEnvVec
from experience import ExperienceSourceFirstLast, unpack_batch
from common import RewardTracker

import numpy as np 
import torch.optim as optim
import torch as T
import gym

from itertools import count
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple

TotalReward = namedtuple("TotalReward", field_names="reward")

def data_func(net, batch_size, entropy_beta, env_name, n_envs,
              gamma, reward_steps, device, train_queue):

    env = GymEnvVec(env_name, n_envs)
    agent = Agent(net, batch_size, entropy_beta)
    exp_source = ExperienceSourceFirstLast(env, agent, gamma, reward_steps)

    for exp in exp_source:
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            train_queue.put(TotalReward(reward=np.mean(new_rewards)))
        train_queue.put(exp)

            


def main():
    
    HYPERPARAMS = {
        "breakout": {
            #"env_name": "BreakoutNoFrameskip-v4",
            "env_name": "PongNoFrameskip-v4",
            "gamma": 0.99, 
            "learning_rate": 0.003,
            "entropy_beta": 0.03,
            "batch_size": 128,
            "n_envs": 15,
            "process_count": 4,
            "reward_steps": 4,
            "stop_reward": 500,
            "adam_eps": 1e-3,
        }
    }

    params = HYPERPARAMS["breakout"]

    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    
    writer = SummaryWriter("run")
    env = GymEnvVec(params["env_name"], params["n_envs"])

    net = A2C(env.envs[0].observation_space.shape, env.envs[0].action_space.n)
    optimizer = optim.Adam(net.parameters(), lr=params["learning_rate"], 
                           eps=params["adam_eps"])

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

            if step >= 10000:
                break
            
            # Output the tuple (batch_states, batch_actions, batch_qvals)
            batch_args = unpack_batch(batch, net, params["gamma"], params["reward_steps"], device=device)
            batch.clear()

            agent.learn(step, *batch_args, optimizer)


if __name__ == "__main__":
    main()