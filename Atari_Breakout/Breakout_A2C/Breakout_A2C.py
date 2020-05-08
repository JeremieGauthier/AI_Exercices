from model import A2C
from agent import Agent
from wrappers import make_env
from utils import QVals, ExperienceSource

import numpy as np 
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch as T
import ptan
import gym

from itertools import count
from collections import deque
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
        }
    }

    params = HYPERPARAMS["breakout"]

    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    env = make_env(params["env_name"])
    writer = SummaryWriter("run")

    net = A2C(env.observation_space.shape, env.action_space.n)
    optimizer = optim.Adam(net.parameters(), lr=params["learning_rate"])

    agent = Agent(net, params["batch_size"], params["entropy_beta"], params["accumulation_steps"])
    exp_source = ExperienceSource(env, agent)
    
    qvals = QVals()
    batch_states, batch_actions, batch_rewards = [], [], []
    scores = []

    for episode in count():
        done = False
        score = 0
        
        qvals.reset()
        optimizer.zero_grad()

        for step, exp in enumerate(exp_source):
            if exp.done:
                break

            batch_states.append(exp.state)
            batch_actions.append(exp.action)
            batch_rewards.append(exp.reward)

            score += exp.reward

            if len(batch_states) < params["batch_size"]:
                continue 

            batch_states_ts = T.FloatTensor(batch_states).to(device)
            batch_actions_ts = T.LongTensor(batch_actions).to(device)
            batch_qvals_ts = T.FloatTensor(qvals.calc_qvals(batch_rewards, params["gamma"])).to(device)
            
            agent.learn(step, batch_states_ts, batch_actions_ts, batch_qvals_ts, optimizer)

            batch_states.clear()
            batch_actions.clear()
            batch_rewards.clear()

        scores.append(score)
        mean_score = np.array(scores[-100:])
        mean_score = np.mean(mean_score)

        writer.add_scalar("score", score, episode) 
        writer.add_scalar("mean_score", mean_score, episode) 

        print("episode :%d, score :%.3f, mean_score :%.3f" % (episode, score, mean_score))