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
            "env_name": "CartPole-v0",
            "gamma": 0.99,
            "learning_rate": 0.001,
            "entropy_beta": 0.01,
            "batch_size": 8,
            "reward_steps": 10
        }
    }

    params = HYPERPARAMS["cartpole"]

    env = gym.make(params["env_name"])
    writer = SummaryWriter("run")

    PG_network = model.PGN(env.observation_space.shape, env.action_space.n)
    print(PG_network)

    agent = ptan.agent.PolicyAgent(PG_network, preprocessor=ptan.agent.float32_preprocessor,
                                    apply_softmax=True)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, 
                            gamma=params["gamma"], steps_count=params["reward_steps"])
    
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
        batch_actions_ts = torch.LongTensor(batch_actions)
        batch_scales_ts = torch.FloatTensor(batch_scales)

        optimizer.zero_grad()
        logits= PG_network(batch_states_ts)
        log_prob = F.log_softmax(logits, dim=1)
        log_prob_actions = batch_scales_ts * log_prob[range(params["batch_size"]), batch_actions_ts]
        loss_policy = -log_prob_actions.mean()

        prob = F.softmax(logits, dim=1)
        entropy = -(prob * log_prob).sum(dim=1).mean()
        entropy_loss = params["entropy_beta"] * entropy
        loss = loss_policy - entropy_loss

        loss.backward()
        optimizer.step()
        
        #Kullback-Leibler Divergence
        new_logits = PG_network(batch_states_ts)
        new_prob = F.softmax(new_logits, dim=1)
        kl_divergence = -((new_prob / prob).log() * prob).sum(dim=1).mean()
        writer.add_scalar("kl", kl_divergence, step)

        grad_max = 0.0
        grad_means = 0.0
        grad_count = 0
        for p in PG_network.parameters():
            grad_max = max(grad_max, p.grad.abs().max().item())
            grad_means += (p.grad ** 2).mean().sqrt().item()
            grad_count += 1

        writer.add_scalar("baseline", baseline, step)
        writer.add_scalar("entropy", entropy.item(), step)
        writer.add_scalar("batch_scales", np.mean(batch_scales), step)
        writer.add_scalar("loss_entropy", entropy_loss.item(), step)
        writer.add_scalar("loss_policy", loss_policy.item(), step)
        writer.add_scalar("loss_total", loss.item(), step)
        writer.add_scalar("grad_l2", grad_means / grad_count, step)
        writer.add_scalar("grad_max", grad_max, step)

        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()
