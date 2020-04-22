import model

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import ptan
import gym

from torch.utils.tensorboard import SummaryWriter


def calc_qvals(rewards, gamma):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= gamma
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))


if __name__ == "__main__":

    HYPERPARAMS = {
        "cartpole": {
            "gamma": 0.99,
            "learning_rate": 0.01,
            "episodes_to_train": 4
        }
    }

    params = HYPERPARAMS["cartpole"]

    env = gym.make("CartPole-v0")
    writer = SummaryWriter("run")

    PG_network = model.PGN(env.observation_space.shape, env.action_space.n)

    agent = ptan.agent.PolicyAgent(PG_network, preprocessor=ptan.agent.float32_preprocessor,
                                    apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, 
                                    gamma=params["gamma"])

    optimizer = optim.Adam(PG_network.parameters(), lr=params["learning_rate"])

    #########################Initialise the variables###################
    total_rewards = []
    step = 0
    completed_episodes = 0 #Number of completed episodes
    batch_episodes = 0 
    current_rewards = [] #Local rewards for the currently-played episode
    batch_states, batch_actions, batch_qvals = [], [], []
    ####################################################################

    for step, exp in enumerate(exp_source):
        
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        current_rewards.append(exp.reward)

        if exp.last_state is None:
            batch_qvals.extend(calc_qvals(current_rewards, params["gamma"]))
            current_rewards.clear()
            batch_episodes += 1

        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            completed_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
        
            ###################### Track New Rewards ######################
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, reward_100: %6.2f, episodes: %d" %
                    (step, reward, mean_rewards, completed_episodes))
            writer.add_scalar("reward", reward, step)
            writer.add_scalar("reward_100", mean_rewards, step)
            if mean_rewards > 195:
                print("Solved in %d steps and %d episodes!" % 
                        (step, completed_episodes))
                break
            ###############################################################

        if batch_episodes < params["episodes_to_train"]:
            continue

        optimizer.zero_grad()
        states_ts = torch.FloatTensor(batch_states)
        batch_actions_ts = torch.LongTensor(batch_actions)
        batch_qvals_ts = torch.FloatTensor(batch_qvals)

        logits = PG_network(states_ts)
        log_prob = F.log_softmax(logits, dim=1)
        log_prob_actions = batch_qvals_ts * log_prob[range(len(batch_states)), batch_actions_ts]
        loss = -log_prob_actions.mean() #Why taking the mean?

        loss.backward()
        optimizer.step()

        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()
    
    writer.close


    
