import torch.nn.functional as F 
import torch as T

class Agent():
    def __init__(self, network, batch_size, entropy_beta, accumulation_steps):
        self.network = network
        self.batch_size = batch_size
        self.entropy_beta = entropy_beta
        self.accumulation_steps = accumulation_steps

    def choose_action(self, state):
        probabilities  = F.softmax(self.network(state)[0])
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()

        return action.item()
    
    def learn(self, step, batch_states, batch_actions, batch_qvals, optimizer):
        critic_values = self.network(batch_states)[1].squeeze()
        delta = batch_qvals - critic_values

        logits = self.network(batch_states)[0]
        log_probs = F.log_softmax(logits, dim=1)
        log_prob_actions = delta * log_probs[range(self.batch_size), batch_actions]

        probs= F.softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()

        entropy_loss = -self.entropy_beta * entropy
        actor_loss = -log_prob_actions
        critic_loss = delta**2
        
        loss  = actor_loss + critic_loss + entropy_loss
        loss.mean().backward()

        # if step % self.accumulation_steps:
        optimizer.step()
        optimizer.zero_grad()