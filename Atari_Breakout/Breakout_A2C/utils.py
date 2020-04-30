import torch.nn.functional as F 
import torch as T

class Agent():
    def __init__(self, network):
        self.network = network

    def choose_action(self, state):
        probabilities  = F.softmax(self.network(state)[0])
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()

        return action.item()