import math
import torch
import numpy as np
import torch.nn.functional as F 
import torch.nn.utils as nn_utils

def calc_logprob(mu, var, actions):
    term_1 = -((mu - actions) ** 2) / (2 * var.clamp(min=1e-3))
    term_2 = - torch.log(torch.sqrt(2 * math.pi * var))
    return term_1 + term_2

class Agent():
    def __init__(self, network, batch_size, entropy_beta):
        self.network = network
        self.batch_size = batch_size
        self.entropy_beta = entropy_beta

    def choose_action(self, state):
        mu, var = self.network(state)[:-1]
        action_probs = torch.distributions.Normal(mu, torch.sqrt(var))
        probs= action_probs.sample()
        #actions_v = torch.tanh(probs) #Using np.clip() is also possible between -1 and 1
        actions_v = np.clip(probs, -1, 1)
        actions = actions_v.squeeze(dim=0)
        actions = actions.data.cpu().numpy()
        
        return actions
    
    def learn(self, step, batch_states, batch_actions, batch_qvals, optimizer):

        mu, var, critic_values = self.network(batch_states)

        #Critic Loss
        critic_loss = F.mse_loss(critic_values.squeeze(-1), batch_qvals)

        #Actor Loss
        adv = batch_qvals - critic_values.squeeze(-1).detach()
        log_prob_actions = adv.unsqueeze(dim=1) * calc_logprob(mu, var, batch_actions)
        actor_loss = -log_prob_actions.mean()

        #Entropy Loss
        entropy_loss = self.entropy_beta * (-(torch.log(2 * math.pi * var) +1 ) / 2).mean()
        
        #Backpropagate
        loss  = critic_loss + entropy_loss + actor_loss
        loss.backward()
        
        #Update the network
        optimizer.step()

        # #Total Loss
        # loss += actor_loss
        
        # #Get the gradients to plot it
        # grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
        #                                 for p in self.network.parameters()
        #                                 if p.grad is not None])