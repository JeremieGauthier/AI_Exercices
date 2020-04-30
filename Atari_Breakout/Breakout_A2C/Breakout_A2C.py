from model import A2C
from wrappers import make_env
from utils import Agent

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
            "reward_steps": 10,
            "entropy_beta": 0.001,
            "batch_size": 8,
            "baseline_step": 1000000
        }
    }

    params = HYPERPARAMS["breakout"]

    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    env = make_env(params["env_name"])
    writer = SummaryWriter("run")

    net = A2C(env.observation_space.shape, env.action_space.n)
    optimizer = optim.Adam(net.parameters(), lr=params["learning_rate"])

    agent = Agent(net)
    
    eps_states, eps_actions, eps_rewards = [], [], []
    scores = []

    for episode in count():
        done = False
        score = 0
        q_val=0
        
        state = env.reset()

        while not done:
            import ipdb; ipdb.set_trace()

            action = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)

            eps_states.append(state_)
            eps_actions.append(action)
            eps_rewards.append(reward)

            state = state_
            score += reward

            if done: 
                R=0
                len_episode = len(eps_actions)

                optimizer.zero_grad()

                eps_states_ts = T.FloatTensor(eps_states).to(device)
                eps_actions_ts = T.LongTensor(eps_actions).to(device)
                eps_rewards_ts = T.FloatTensor(eps_rewards).to(device)

                for step in range(len_episode):
                    time = len_episode - step - 1

                    R *= params["gamma"]
                    R *= eps_rewards[time]

                    critic_values_time_ts = net(eps_states_ts[time])[1].squeeze()

                    delta = R - critic_values_time_ts

                    logits = net(eps_states_ts[time])[0].squeeze()
                    log_probs = F.log_softmax(logits, dim=0)
                    log_prob_actions = log_probs[eps_actions_ts[time]] * delta

                    actor_loss = -log_prob_actions
                    critic_loss = delta**2

                    (actor_loss + critic_loss).backward()

                    optimizer.step()

                eps_states.clear()
                eps_actions.clear()
                eps_rewards.clear()

        scores.append(score)
        mean_score = np.array(scores[-100:])
        mean_score = np.mean(mean_score)

        writer.add_scalar("score", score, episode) 
        writer.add_scalar("mean_score", mean_score, episode) 

        if episode % 1 == 0:
            print("episode :%d, score :%.3f, mean_score :%.3f" % (episode, score, mean_score))