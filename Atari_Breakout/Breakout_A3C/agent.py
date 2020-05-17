import torch.nn.functional as F 
import torch.nn.utils as nn_utils
import torch as T

class Agent():
    def __init__(self, network, batch_size, entropy_beta):
        self.network = network
        self.batch_size = batch_size
        self.entropy_beta = entropy_beta

    def choose_action(self, state):
        probabilities  = F.softmax(self.network(state)[0])
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()

        return action.item()
    
    def learn(self, step, batch_states, batch_actions, batch_qvals, optimizer):
        logits, critic_values = self.network(batch_states)

        #Critic Loss
        critic_loss = F.mse_loss(batch_qvals, critic_values.squeeze())

        #Actor Loss
        log_probs = F.log_softmax(logits, dim=1)
        adv = batch_qvals - critic_values.squeeze()
        import ipdb; ipdb.set_trace()
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
        optimizer.zero_grad()