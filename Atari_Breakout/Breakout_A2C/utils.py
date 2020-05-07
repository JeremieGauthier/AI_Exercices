from collections import namedtuple

class QVals():
    def __init__(self):
        self.R = 0.0

    def reset(self):
        self.R = 0.0

    def calc_qvals(self, rewards, gamma):
        res = []
        for r in reversed(rewards):
            self.R *= gamma
            self.R += r
            res.append(self.R)
        return list(reversed(res))

    
Experience = namedtuple("Experience", ("state", "action", "reward", "done"))


class ExperienceSource():
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        
    def __iter__(self):

        obs = self.env.reset()
        while True:

            action = self.agent.choose_action(obs)
            state, reward, done, _ = self.env.step(action)

            yield Experience(state, action, reward, done)
            