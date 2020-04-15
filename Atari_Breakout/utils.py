import numpy as np
import time

class RewardTracker():
    def __init__(self, writer, params):
        self.writer = writer #tensorboard.SummaryWriter
        self.stop_reward = params["stop_reward"]

    def __enter__(self):
        self.start = time.time()
        self.timestep_frame = 0
        self.total_rewards = []

        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, current_frame, epsilon=None):

        self.total_rewards.append(reward)
        speed = (current_frame - self.timestep_frame)\
            /(time.time() - self.start)
        self.timestep_frame = current_frame
        self.start = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])


        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s %s" % (
            current_frame, len(self.total_rewards), mean_reward, 
            speed, epsilon_str))

        if epsilon is not None: 
            self.writer.add_scalar("epsilon", epsilon, current_frame)
            self.writer.add_scalar("speed", speed, current_frame)
            self.writer.add_scalar("reward_100", mean_reward, current_frame)
            self.writer.add_scalar("reward", reward, current_frame)
        
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % current_frame)
            return True

        return  False

        