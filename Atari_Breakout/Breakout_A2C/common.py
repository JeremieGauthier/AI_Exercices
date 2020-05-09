import time
import numpy as np


class RewardTracker():
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward
    
    def __enter__(self):
        self.scores = []
        return self
    
    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, step):
        self.scores.append(reward)
        mean_reward = np.array(self.scores[-100:])
        mean_reward = np.mean(mean_reward)

        self.writer.add_scalar("score", reward, step) 
        self.writer.add_scalar("mean_score", mean_reward, step) 

        if step % 100 == 0:
            print("episode :%d, score :%.3f, mean_score :%.3f" % (step, reward, mean_reward))

        if mean_reward > self.stop_reward:
            return True
        return False