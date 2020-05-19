from model import A2C
from agent import Agent
from wrappers import GymEnvVec
from experience import ExperienceSourceFirstLast, unpack_batch
from common import RewardTracker

import numpy as np 
import torch.optim as optim
import torch.multiprocessing as mp
import torch as T
import ptan
import gym

from itertools import count
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple



TotalReward = namedtuple("TotalReward", field_names="reward")

def data_func(net, device, train_queue, batch_size, entropy_beta, 
              env_name, n_envs, gamma, reward_steps, **kwargs):

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

    make_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make(params["env_name"]))

    env = make_env()
    net = A2C(env.observation_space.shape, env.action_space.n)
    net.share_memory()
    
    optimizer = optim.Adam(net.parameters(), lr=params["learning_rate"], eps=params["adam_eps"])

    train_queue = mp.Queue(maxsize=params["process_count"])
    data_proc_list = []
    
    for _ in range(params["process_count"]):
        data_proc = mp.Process(target=data_func, 
                               args=(net, device, train_queue), 
                               kwargs={**params})
        data_proc.start()
        data_proc_list.append(data_proc)
    


if __name__ == "__main__":
    main()