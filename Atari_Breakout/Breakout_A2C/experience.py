import torch as T
import numpy as np

from collections import namedtuple, deque

Experience = namedtuple("Experience", ("state", "action", "reward", "done"))

class ExperienceSource():
    def __init__(self, env, agent, reward_steps):
        self.env = env
        self.agent = agent
        #self.reward_steps = reward_steps
        #Here N-1 = 1
        self.reward_steps = 1
        self.total_reward = []
        
    def __iter__(self):
        current_rewards = [0.0] * len(self.env.envs)
        histories = [deque(maxlen=self.reward_steps)] * len(self.env.envs)

        states = self.env.reset()
        while True:

            for idx, env in enumerate(self.env.envs):
                action = self.agent.choose_action(np.array(states[idx], copy=False))
                state, reward, done, _ = env.step(action)
                
                current_rewards[idx] += reward
                histories[idx].append(Experience(state, action, reward, done))

                if len(histories[idx]) == self.reward_steps:
                    yield tuple(histories[idx])

                if done: 
                    self.total_reward.append(current_rewards[idx])
                    yield tuple(histories[idx])

                    state = env.reset()
                    current_rewards[idx] = 0.0
                    histories[idx].clear()
                
    
    def pop_total_reward(self):
        """
        This part is only used to track the total reward.
        If new_reward=True, it means the episode is done
        """
        
        r = self.total_reward
        if r:
            self.total_reward = []
        return r


ExperienceFirstLast = namedtuple("ExperienceFirstLast", ("state", "action", "reward", "last_state"))


class ExperienceSourceFirstLast(ExperienceSource):
    """
    This is a wrapper around ExperienceSource to prevent storing full trajectory in replay buffer when we need
    only first and last states. For every trajectory piece it calculates discounted reward and emits only first
    and last states and action taken in the first state.

    If we have partial trajectory at the end of episode, last_state will be None
    """
    def __init__(self, env, agent, gamma, reward_steps):
        super(ExperienceSourceFirstLast, self).__init__(env, agent, reward_steps+1)

        self.gamma = gamma
        self.steps = reward_steps
    
    def __iter__(self):
        for exp in super(ExperienceSourceFirstLast, self).__iter__():
            if exp[-1].done and len(exp) <= self.steps:
                last_state = None
                elems = exp
            else:
                last_state = exp[-1].state
                elems = exp[:-1]
            
            total_reward = 0.0
            for elem in reversed(elems):
                total_reward *= self.gamma
                total_reward += elem.reward

            yield ExperienceFirstLast(exp[0].state, action=exp[0].action,
                                      reward=total_reward, last_state=last_state)


def unpack_batch(batch, net, gamma, reward_steps, device='cpu'):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """

    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))

    states_v = T.FloatTensor(np.array(states, copy=False)).to(device)
    actions_t = T.LongTensor(actions).to(device)
    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    
    if not_done_idx:
        last_states_v = T.FloatTensor(np.array(last_states, copy=False)).to(device)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        last_vals_np *= gamma ** reward_steps
        rewards_np[not_done_idx] += last_vals_np

    ref_vals_v = T.FloatTensor(rewards_np).to(device)
    return states_v, actions_t, ref_vals_v
