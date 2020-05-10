import torch as T
import numpy as np

from collections import namedtuple, deque


Experience = namedtuple("Experience", ("state", "action", "reward", "done"))


class ExperienceSource():
    def __init__(self, env, agent, reward_steps):
        self.env = env
        self.agent = agent
        self.reward_steps = reward_steps
        self.total_reward = []
        
    def __iter__(self):
        current_reward = 0.0
        state = self.env.reset()
        while True:
            
            history = deque(maxlen=self.reward_steps)
            action = self.agent.choose_action(state)
            state, reward, done, _ = self.env.step(action)

            current_reward += reward

            history.append(Experience(state, action, reward, done))

            if len(history) == self.reward_steps:
                yield tuple(history)

            if done: 
                self.total_reward.append(current_reward)
                yield tuple(history)

                state = self.env.reset()
                current_reward = 0.0
                
    
    def pop_total_reward(self):
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
        self.reward_steps = reward_steps
    
    def __iter__(self):
        for exp in super(ExperienceSourceFirstLast, self).__iter__():
            if exp[-1].done and len(exp) <= self.reward_steps:
                last_state = None
                elems = exp
            else:
                last_state = exp[-1].state
                elems = exp[:-1]
            
            total_reward = 0.0
            for elem in reversed(elems):
                total_reward *= self.gamma
                total_reward *= elem.reward

            yield ExperienceFirstLast(exp[0].state, action=exp[0].action,
                                      reward=total_reward, last_state=last_state)


def unpack_batch(batch, net, gamma, reward_steps, device):
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(exp.action)
        rewards.append(exp.reward)

        #if last_state=None, it means the game ends up
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))

    states_ts = T.FloatTensor(states).to(device)
    actions_ts = T.LongTensor(actions).to(device)
    rewards_ts = T.FloatTensor(rewards).to(device)

    if not_done_idx:
        last_states_v = T.FloatTensor(last_states).to(device)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_ts[not_done_idx] += gamma**reward_steps * last_vals_np
    ref_vals_v = T.FloatTensor(rewards_ts).to(device)

    return states_ts, actions_ts, ref_vals_v
