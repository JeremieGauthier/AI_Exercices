import torch.nn as nn

class PGN(nn.Module):
    def __init__(self, input_size, num_actions):
        super(PGN, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(in_features=input_size[0], out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_actions)
        )
    
    def forward(self, state):
        return self.linear(state.float())