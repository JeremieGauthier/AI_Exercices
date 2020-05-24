import torch
import torch.nn as nn

HID_SIZE = 128

class ModelA2C(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelA2C, self).__init__()
        
        self.base = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
        )

        #Tanh() allows us to have a mean value between 0 and 1
        self.mu = nn.Sequential(
           nn.Linear(HID_SIZE, act_size),
           nn.Tanh(),
        )

        #Softplus() allows us to have a positive variance
        self.var = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Softplus(),
        )

        self.value = nn.Linear(HID_SIZE, 1)
    
    def forward(self, state):
        base = self.base(state)
        return self.mu(base), self.var(base), self.value(base)
