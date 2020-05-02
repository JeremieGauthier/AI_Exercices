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

class QVals():
    def __init__(self):
        self.R = 0.0

    def reset(self):
        self.R = 0.0

    def calc_qvals(self, rewards, gamma):
        res = []
        for r in reversed(rewards):
            self.R *= gamma
            self.R += r
            res.append(self.R)
        return list(reversed(res))