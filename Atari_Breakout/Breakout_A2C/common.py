import time
import numpy as np


class RewardTracker():
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward
    
    def __enter__(self):
        self.total_rewards = []
        return self
    
    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, step):
        self.total_rewards.append(reward)
        mean_reward = np.array(self.total_rewards[-100:])
        mean_reward = np.mean(mean_reward)

        self.writer.add_scalar("score", reward, step) 
        self.writer.add_scalar("mean_score", mean_reward, step) 

        print("done :%d, game :%d, score :%.3f, mean_score :%.3f" % 
                  (step, len(self.total_rewards), reward, mean_reward))

        if mean_reward > self.stop_reward:
            return True
        return False