import time
import numpy as np


class RewardTracker():
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward
    
    def __enter__(self):
        self.total_rewards = []
        self.start_time = time.time()
        self.start_frame = 0
        return self
    
    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame):
        self.total_rewards.append(reward)
        mean_reward = np.array(self.total_rewards[-100:])
        mean_reward = np.mean(mean_reward)

        speed = (frame - self.start_frame) / (time.time() - self.start_time)
        self.start_frame = frame
        self.start_time = time.time()

        self.writer.add_scalar("score", reward, frame) 
        self.writer.add_scalar("mean_score", mean_reward, frame) 

        num_games = len(self.total_rewards)
        if num_games % 1 == 0:
            print("done :%d, game :%d, reward :%.3f, mean_reward :%.3f, speed :%.3f" % 
                    (frame+1, len(self.total_rewards), reward, mean_reward, speed))

        if mean_reward > self.stop_reward:
            return True
        return False