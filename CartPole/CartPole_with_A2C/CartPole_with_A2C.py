from actor_critic_discrete import Agent

from gym import wrappers
from torch.utils.tensorboard import SummaryWriter

import numpy as np 
import gym

if __name__ == "__main__":
    agent = Agent(alpha = 0.00001, beta=0.0005, input_dims=[4], n_actions=2,
                l1_size=32, l2_size=32)

    env = gym.make("CartPole-v1")
    writer = SummaryWriter("run")

    num_episodes = 2500
    for step in range(num_episodes):
        score=0
        done=False
        state = env.reset()
        while not done:
            action = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            score += reward
            agent.learn(state, reward, state_, done)
            state = state_
        writer.add_scalar("score", score, step) 
        print("episode :%d, score :%.3f" % (step, score))
