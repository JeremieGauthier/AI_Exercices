from actor_critic_discrete import GenericNetwork

from gym import wrappers
from torch.utils.tensorboard import SummaryWriter
from itertools import count

import torch.nn.functional as F
import numpy as np 
import torch as T
import gym

def choose_action(network, states):
    probabilities  = F.softmax(network.forward(states))
    action_probs = T.distributions.Categorical(probabilities)
    action = action_probs.sample()

    return action.item()


if __name__ == "__main__":

    env = gym.make("CartPole-v1")
    writer = SummaryWriter("run")

    device = T.device("cuda" if T.cuda.is_available() else "cpu")

    actor = GenericNetwork(0.00001, 4, 32, 32, 2)
    critic = GenericNetwork(0.0005, 4, 32, 2, 1)

    batch_states, batch_actions, batch_rewards = [], [], []

    batch_size = 8
    for step in count():
        score=0
        done=False
        state = env.reset()

        while not done:
            action = choose_action(actor, state)
            state_, reward, done, _ = env.step(action)

            batch_actions.append(action)
            batch_rewards.append(reward)
            batch_states.append(state_)

            score += reward            
            state = state_

            if len(batch_states) < batch_size:
                continue
            
            batch_states_ts = T.FloatTensor(batch_states).to(device=device)
            batch_actions_ts = T.LongTensor(batch_actions).to(device=device)
            batch_rewards_ts = T.FloatTensor(batch_rewards).to(device=device)

            actor.optimizer.zero_grad()
            critic.optimizer.zero_grad()

            critic_values_ts = critic.forward(batch_states)
            # critic_values_ = critic.forward(new_states)

            # delta = ((rewards + gamma * critic_values_ * (1 - int(dones))) - critic_values)
            delta = batch_rewards_ts - critic_values_ts

            logits = actor(batch_states_ts)
            log_prob = F.log_softmax(logits, dim=1)
            log_prob_actions = log_prob[range(batch_size), batch_actions_ts] * delta

            actor_loss = -log_prob_actions.mean()
            critic_loss = (delta**2).mean()

            (actor_loss + critic_loss).backward()

            actor.optimizer.step()
            critic.optimizer.step()

            batch_states.clear()
            batch_actions.clear()
            batch_rewards.clear()

        writer.add_scalar("score", score, step) 
        print("episode :%d, score :%.3f" % (step, score))
