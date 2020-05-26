import torch.nn.functional as F 
import torch.nn.utils as nn_utils
import torch as T
import numpy as np

class Agent():
    def __init__(self, network, batch_size, entropy_beta):
        self.network = network
        self.batch_size = batch_size
        self.entropy_beta = entropy_beta

    def choose_action(self, state):
        probabilities  = F.softmax(self.network(state)[0], dim=-1)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        import ipdb; ipdb.set_trace()

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

        #Total Loss
        loss += actor_loss
        
        #Get the gradients to plot it
        grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                        for p in self.network.parameters()
                                        if p.grad is not None])

        return locals()