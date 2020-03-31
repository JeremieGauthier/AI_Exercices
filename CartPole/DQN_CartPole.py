import gym
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torch as T 

import matplotlib.pyplot as plt 


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

class LinearDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(LinearDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)

        return actions

class Agent():
    def __init__(self, input_dims, batch_size, n_actions, lr, replace, 
                gamma=0.99, epsilon=1.0, eps_dec=1e-5, eps_min=0.01, 
                max_mem_size=1000000):
        
        self.lr = lr
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [i for i in range(self.n_actions)]
        self.memory_size = max_mem_size
        self.memory_counter = 0
        self.replace = replace

        self.Q_eval = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)
        self.Q_next = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)

        self.state_memory = np.zeros((self.memory_size, *self.input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.memory_size, *self.input_dims), dtype=np.float32)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.action_memory = np.zeros(self.memory_size, dtype=np.int32)
        self.terminal_state = np.zeros(self.memory_size, dtype=np.bool)
    
    def replace_target_network(self):
        self.Q_next.load_state_dict(self.Q_eval.state_dict())
        self.Q_next.eval()


    def store_transition(self, state, action, reward, state_, done):
        index = self.memory_counter % self.memory_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_state[index] = done

        self.memory_counter += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(observation, dtype=T.float).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                if self.epsilon > self.eps_min else self.eps_min
    
    def learn(self, state, action, reward, state_):

        if self.memory_counter > self.batch_size:
            self.Q_eval.optimizer.zero_grad()

            max_memory = min(self.memory_counter, self.memory_size)
            batch = np.random.choice(max_memory, self.batch_size, replace=False)
            batch_index = np.arange(self.batch_size, dtype=np.int32)

            states = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
            actions = T.tensor(self.action_memory[batch], dtype=T.float32).to(self.Q_eval.device)
            rewards = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
            states_ = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
            terminal = T.tensor(self.terminal_state[batch]).to(self.Q_eval.device)

            q_pred = self.Q_eval.forward(states)[batch_index, actions.type(T.LongTensor)]
            q_next = self.Q_next.forward(states_)
            q_next[terminal] = 0.0

            q_target = reward + self.gamma * T.max(q_next, dim=1)[0]

            loss = self.Q_eval.loss(q_target, q_pred).to(self.Q_eval.device)
            loss.backward()
            self.Q_eval.optimizer.step()
            self.decrement_epsilon()

if __name__ == "__main__":

    env = gym.make("CartPole-v1")
    n_games = 8000
    scores, eps_history = [], []
   
    agent = Agent(batch_size=64, input_dims=env.observation_space.shape,
                    replace=10, n_actions=env.action_space.n, lr=0.0001)
    
    for i in range(n_games):
        score = 0
        done = False
        obs = env.reset()

        if i % agent.replace == 0:
            agent.replace_target_network()

        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(obs, action, reward, obs_, done)
            agent.learn(obs, action, reward, obs_)
            obs = obs_
        scores.append(score)
        eps_history.append(agent.epsilon)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print("episode", i, "score %.1f average score %.1f epsilon %.2f" %
               (score, avg_score, agent.epsilon))

    filename = 'cartpole_naive_dqn.png'
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores, eps_history, filename)

