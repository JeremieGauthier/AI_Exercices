import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torch as T
import numpy as np 

class GenericNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(GenericNetwork, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device)
        output = F.relu(self.fc1(state))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output


class Agent():
    def __init__(self, alpha, beta, input_dims, gamma=0.99, 
                l1_size=256, l2_size=256, n_actions=2):
        self.gamma = gamma
        self.log_probs = None
        self.actor = GenericNetwork(alpha, input_dims, l1_size, l2_size, 
                                    n_actions)
        self.critic = GenericNetwork(beta, input_dims, l1_size, l2_size, 
                                    n_actions=1)
        
    def choose_action(self, observation):
        import ipdb; ipdb.set_trace()
        probabilities  = F.softmax(self.actor.forward(observation))
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        self.log_probs = action_probs.log_prob(action)

        return action.item()
    
    def learn(self, state, reward, new_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        critic_value = self.critic.forward(state)
        critic_value_ = self.critic.forward(new_state)

        delta = ((reward + self.gamma * critic_value_ * (1 - int(done))) - critic_value)

        actor_loss = -1*self.log_probs * delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()

        self.actor.optimizer.step()
        self.critic.optimizer.step()
