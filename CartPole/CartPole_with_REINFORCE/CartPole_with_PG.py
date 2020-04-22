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
    res=[]
    sum_r = 0
    for r in reversed(rewards):
        sum_r *= gamma
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))

if __name__ == "__main__":
    HYPERPARAMS = {
        "cartpole": {
            "env_nama": "CartPole-v0",
            "gamma": 0.99,
            "learning_rate": 0.001,
            "entropy_beta": 0.01,
            "batch_size": 8,
            "rewards_step": 10
        }
    }

    params = HYPERPARAMS["cartpole"]

    env = gym.make(params["env_name"])
    writer = SummaryWriter("run")

    PG_network = model.PGN(env.observation_space.shape, env.action_space.n)
    print(PG_network)

    agent = ptan.agent.PolicyAgent(PG_network, preprocessor=ptan.agent.float32_preprocessor,
                                    apply_softmax=True)

    exp_source = ptan.experience.ExperienceFirstLast(env, agent, 
                            gamma=params["gamma"], step_count=params["reward_steps"])
    
    optimizer = optim.Adam(PG_network.parameters(), lr=params["learning_rate"])


    #########################Initialise the variables###################
    total_rewards = []
    step_rewards = []
    step = 0
    completed_episodes = 0 #Number of completed episodes
    reward_sum = 0.0

    batch_states, batch_actions, batch_scales = [], [], []
    ####################################################################

    for step, exp in enumerate(exp_source):
        reward_sum += exp.reward
        baseline = reward_sum / (step + 1) #Mean Reward
        writer.add_scalar("baseline", baseline, step)
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        batch_scales.append(exp.reward - baseline)

        # PG might have high variance in complex environments.
        # Otherwise the training process can become unstable.
        # The usual way is to subtract baseline from the Q

        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            completed_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, reward_100: %6.2f, episodes: %d" %
                    (step, reward, mean_rewards, completed_episodes))
            writer.add_scalar("reward", reward, step)
            writer.add_scalar("reward_100", mean_rewards, step)
            writer.add_scalar("episodes", completed_episodes, step)
            if mean_rewards > 195:
                print("Solved in %d steps and %d episodes!" % (step, completed_episodes))
                break

        if len(batch_states) < params["batch_size"]:
            continue

        batch_states_ts = torch.FloatTensor(batch_states)
        batch_actions_ts = torch.IntTensor(batch_actions)
        batch_scales_ts = torch.FloatTensor(batch_scales)

        optimizer.zero_grad()
        logits= PG_network(batch_states_ts)
        log_prob = F.log_softmax(logits, dim=1)
        log_prob_actions = batch_scales_ts * log_prob_v[range(params["batch_size"]), batch_actions_ts]
        loss_policy = -log_prob_actions.mean()

