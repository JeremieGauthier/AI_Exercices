import torch.nn.functional as F 
import torch.nn.utils as nn_utils
import torch
import numpy as np
import math

class Agent():
    def __init__(self, network, batch_size, entropy_beta, num_output):
        self.network = network
        self.batch_size = batch_size
        self.entropy_beta = entropy_beta
        self.num_output = num_output

    def choose_action(self, state):
        #TODO: Might want to check again
        mu, sigma = self.network()[:-1]
        action_probs = torch.distributions.Normal(mu, math.sqrt(sigma))
        probs= action_probs.sample(sample_shape=torch.Size(self.num_output))
        action = torch.tanh(probs)

        return action.item()
    
    def learn(self, step, batch_states, batch_actions, batch_qvals, optimizer):
        logits, critic_values = self.network(batch_states)

        #Critic Loss
        critic_loss = F.mse_loss(critic_values.squeeze(-1), batch_qvals)

        #Actor Loss
        log_probs = F.log_softmax(logits, dim=1)
        adv = batch_qvals - critic_values.squeeze(-1).detach()
        log_prob_actions = adv * log_probs[range(self.batch_size), batch_actions]
        actor_loss = -log_prob_actions.mean()

        #Entropy Loss
        probs= F.softmax(logits, dim=1)
        entropy_loss = self.entropy_beta * (probs * log_probs).sum(dim=1).mean()
        
        #Backpropagate
        actor_loss.backward(retain_graph=True)
        loss  = critic_loss + entropy_loss
        loss.backward()
        
        #Update the network
        nn_utils.clip_grad_norm_(self.network.parameters(), 0.1)
        optimizer.step()

        # #Total Loss
        # loss += actor_loss
        
        # #Get the gradients to plot it
        # grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
        #                                 for p in self.network.parameters()
        #                                 if p.grad is not None])