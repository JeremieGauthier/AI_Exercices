import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(x, scores, epsilons, filename):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Step", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis="x", color="C0")
    ax.tick_params(axis="y", color="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel("Score", color="C1")
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(axis='y', color="C1")

    plt.savefig(filename)

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
        speed = (current_step - self.timestep_frame)/
                    (time.time() - self.start)
        self.timestep_frame = current_frame
        self.start = time.time()
        mean_reward = np.mean(self.total_rewards[:-100])

        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))

        #sys.stdout.flush()

        if epsilon is not None: 
            self.writer.add_scalar("epsilon", epsilon, current_frame)
            self.writer.add_scalar("speed", speed, current_frame)
            self.writer.add_scalar("reward_100", mean_reward, current_frame)
            self.writer.add_scalar("reward", reward, current_frame)
        
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return  False

        