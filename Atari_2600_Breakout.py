import gym
import time

env = gym.make('Breakout-v0')
env.reset()

for _ in range(10000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    time.sleep(0.03)

if done:
    obervaton = env.reset()
env.close()