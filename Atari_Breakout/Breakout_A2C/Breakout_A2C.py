from model import A2C
from wrappers import make_env
from utils import Agent, QVals

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
        }
    }

    params = HYPERPARAMS["breakout"]

    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    env = make_env(params["env_name"])
    writer = SummaryWriter("run")

    net = A2C(env.observation_space.shape, env.action_space.n)
    optimizer = optim.Adam(net.parameters(), lr=params["learning_rate"])

    agent = Agent(net)
    
    qvals = QVals()
    batch_states, batch_actions, batch_rewards = [], [], []
    scores = []

    for episode in count():
        done = False
        score = 0
        
        state = env.reset()
        qvals.reset()

        optimizer.zero_grad()
        # for step in count():
        step=0
        while not done:
            step += 1

            action = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)

            batch_states.append(state)
            batch_actions.append(action)
            batch_rewards.append(reward)

            state = state_
            score += reward

            if len(batch_states) < params["batch_size"]:
                continue 

            batch_states_ts = T.FloatTensor(batch_states).to(device)
            batch_actions_ts = T.LongTensor(batch_actions).to(device)
            batch_qvals_ts = T.FloatTensor(qvals.calc_qvals(batch_rewards, params["gamma"])).to(device)
            
            critic_values = net(batch_states_ts)[1].squeeze()

            delta = batch_qvals_ts - critic_values

            logits = net(batch_states_ts)[0]
            log_probs = F.log_softmax(logits, dim=1)
            log_prob_actions = delta * log_probs[range(params["batch_size"]), batch_actions_ts]

            probs= F.softmax(logits, dim=1)
            entropy = -(probs * log_probs).sum(dim=1).mean()

            entropy_loss = -params["entropy_beta"] * entropy
            actor_loss = -log_prob_actions
            critic_loss = delta**2
            
            loss  = actor_loss + critic_loss + entropy_loss
            loss.mean().backward()

            if step % params["accumulation_steps"] or done:
                optimizer.step()
                optimizer.zero_grad()

            batch_states.clear()
            batch_actions.clear()
            batch_rewards.clear()

        scores.append(score)
        mean_score = np.array(scores[-100:])
        mean_score = np.mean(mean_score)

        writer.add_scalar("score", score, episode) 
        writer.add_scalar("mean_score", mean_score, episode) 

        print("episode :%d, score :%.3f, mean_score :%.3f" % (episode, score, mean_score))