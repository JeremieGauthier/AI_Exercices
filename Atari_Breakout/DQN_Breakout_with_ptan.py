import wrappers
import utils

import gym
import time
import math
import random
import ptan
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

from itertools import count
from collections import namedtuple

Experience = namedtuple("Experience",
                            ("state", "action", "reward", "next_state"))

class DQN(nn.Module):
    def __init__(self, num_actions, lr):
        super(DQN, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=4, stride=2)

        # You have to respect the formula ((W-K+2P/S)+1)
        self.fc = nn.Linear(in_features=32*9*9, out_features=256)
        self.out = nn.Linear(in_features=256, out_features=num_actions)


    def forward(self, state):
        # (1) Hidden Conv. Layer
        self.layer1 = F.relu(self.conv1(state))

        # (2) Hidden Conv. Layer
        self.layer2 = F.relu(self.conv2(self.layer1))
        
        # (3) Hidden Linear Layer
        input_layer3 = self.layer2.reshape(-1, 32*9*9)
        self.layer3 = F.relu(self.fc(input_layer3))

        # (4) Output
        actions = self.out(self.layer3)

        return actions

class EpsilonTracker():
    def __init__(self, epsilon_greedy_selector, params):

        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.eps_start = params["eps_start"]
        self.eps_end = params["eps_end"]
        self.eps_frame = params["eps_frame"]
        self.frame(0)
    
    def frame(self, current_step):
        self.epsilon_greedy_selector.epsilon = \
            max(self.eps_end, self.eps_start - current_step / 
                self.eps_frame)

def extract_tensor(experiences):
    batch = list(zip(*experiences))

    states = torch.tensor(batch[0], dtype=torch.float32) 
    actions = torch.tensor(batch[1], dtype=torch.int16) 
    rewards = torch.tensor(batch[2], dtype=torch.float32) 
    try:
        next_states = torch.tensor(batch[3], dtype=torch.float32) 
    except:
        import ipdb; ipdb.set_trace()
        
    return (states, actions, rewards, next_states)


if __name__ == "__main__":
    HYPERPARAMS={
        "breakout":{
            "env_name" : "Breakout-v0",
            "learning_rate" : 0.001,
            "gamma" : 0.99,
            "eps_start" : 1,
            "eps_end" : 0.01,
            "eps_frame" : 10**5,
            "target_update" : 10,
            "num_episodes" : 1500,
            "batch_size" : 20,
            "capacity" : 100000,
            "max_nb_elements" : 4,
        },
    }

    params = HYPERPARAMS["breakout"]

    scores, eps_history = [], []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = wrappers.make_env("Breakout-v0")

    policy_network = DQN(env.action_space.n, params["learning_rate"]).to(device)
    target_network = ptan.agent.TargetNet(policy_network)
    optimizer = optim.Adam(policy_network.parameters(), lr=params["learning_rate"])

    action_selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params["eps_start"])
    agent = ptan.agent.DQNAgent(policy_network, action_selector, device)
    epsilon_tracker = EpsilonTracker(action_selector, params)

    exp_source  = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params["gamma"], steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params["capacity"])

    current_step = 0

    for episode in range(params["num_episodes"]):
        obs = env.reset()
        state = env.observation(obs)
        if state.shape == None:
            import ipdb; ipdb.set_trace()

        state = torch.tensor(state).unsqueeze(dim=0)

        score = 0
        start = time.time()

        for timestep in range(params["batch_size"]):
            epsilon_tracker.frame(current_step)
            current_step += 1

            action  = agent(state)[0] #agent.__call__() return action, agent_states
            new_state, reward, done, _ = env.step(action)
            new_state = torch.tensor(new_state).unsqueeze(dim=0)
            buffer.populate(1)
            if new_state.shape == None:
                import ipdb; ipdb.set_trace()
            state = new_state

            score += reward

            if len(buffer) >= params["batch_size"]:
                experiences = buffer.sample(params["batch_size"])
                states, actions, rewards, next_states = extract_tensor(experiences)

                batch_index = np.arange(params["batch_size"], dtype=np.int32) 
                current_q_value = policy_network.forward(states)[batch_index, actions.type(torch.LongTensor)]
                next_q_value = target_network.target_model(next_states)
                target_q_value = reward + params["gamma"] * next_q_value.max(1)[0]

                loss = nn.MSELoss()
                loss = loss(target_q_value, current_q_value).to(device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
               break
        

        scores.append(score)
        eps_history.append(action_selector.epsilon)

        if episode % params["target_update"] == 0:
            target_network.sync()
        
        if episode != 0 and episode % 200 == 0:
            save = {'state_dict': policy_network.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(save, "DQN_model_" + str(episode)+ "_" 
                       + str(int(score)) + ".pkl")

        print("episode :", episode, "epsilon :", action_selector.epsilon, "score", score,
                "time :", time.time()-start)

        if episode % 20 == 0:
            avg_score = np.mean(scores[-20:])
            print("episode", episode, "score %.1f average score %.1f epsilon %.2f" %
               (score, avg_score, action_selector.epsilon))
    
        if episode % 100 == 0 and episode != 0:

            filename = 'Atari_Breakout_DQN.png'
            x = [i for i in range(episode+1)]
            utils.plot_learning_curve(x, scores, eps_history, filename)