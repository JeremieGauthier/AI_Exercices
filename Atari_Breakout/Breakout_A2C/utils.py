from collections import namedtuple


Experience = namedtuple("Experience", ("state", "action", "reward", "done"))


class ExperienceSource():
    def __init__(self, env, agent, reward_steps):
        self.env = env
        self.agent = agent
        self.reward_steps = reward_steps
        
    def __iter__(self):

        obs = self.env.reset()
        while True:
            
            history = []
            action = self.agent.choose_action(obs)
            state, reward, done, _ = self.env.step(action)

            history.append(Experience(state, action, reward, done))

            if len(history) == self.reward_steps:
                yield tuple(history)
            
            if done:
                yield tuple(history)
                history.clear()


ExperienceFirstLast = namedtuple("ExperienceFirstLast", ("state", "action", "reward", "last_state"))


class ExperienceSourceFirstLast(ExperienceSource):
    """
    This is a wrapper around ExperienceSource to prevent storing full trajectory in replay buffer when we need
    only first and last states. For every trajectory piece it calculates discounted reward and emits only first
    and last states and action taken in the first state.

    If we have partial trajectory at the end of episode, last_state will be None
    """
    def __init__(self, env, agent, gamma, reward_steps):
        super(ExperienceSourceFirstLast, self).__init__(env, agent, gamma, reward_steps+1)

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
