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


def main():
    
    HYPERPARAMS = {
        "breakout": {
            #"env_name": "BreakoutNoFrameskip-v4",
            "env_name": "PongNoFrameskip-v4",
            "gamma": 0.99, 
            "learning_rate": 0.003,
            "entropy_beta": 0.03,
            "batch_size": 128,
            "accumulation_steps": 10,
            "n_envs": 5, 
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

    agent = Agent(net, params["batch_size"], params["entropy_beta"])
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
            batch_args = unpack_batch(batch, net, params["gamma"], params["reward_steps"], device=device)
            batch.clear()

            optimizer.zero_grad()
            kwargs = agent.learn(step, *batch_args, optimizer)
            
            writer.add_scalar("advantage",       kwargs["adv"].mean(), step)
            writer.add_scalar("values",          kwargs["critic_values"].mean(), step)
            writer.add_scalar("batch_rewards",   kwargs["batch_qvals"].mean(), step)
            writer.add_scalar("loss_entropy",    kwargs["entropy_loss"], step)
            writer.add_scalar("loss_policy",     kwargs["actor_loss"], step)
            writer.add_scalar("loss_value",      kwargs["actor_loss"], step)
            writer.add_scalar("loss_total",      kwargs["loss"], step)
            writer.add_scalar("grad_l2",         np.sqrt(np.mean(np.square(kwargs["grads"]))), step)
            writer.add_scalar("grad_max",        np.max(np.abs(kwargs["grads"])), step)
            writer.add_scalar("grad_var",        np.var(kwargs["grads"]), step)



if __name__ == "__main__":
    main()