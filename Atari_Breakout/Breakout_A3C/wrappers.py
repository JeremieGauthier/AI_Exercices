import ptan
import gym

class GymEnvVec():
    """
    Used to make a vector of OpenAI gym environments
    Having more than one env help to break the correlation and make
    the samples i.i.d
    """
    def __init__(self, env_name, n_envs, seed=0):
        build_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make(env_name))
        self.envs = [build_env() for _ in range(n_envs)]
        [env.seed(seed + 10 * i) for i, env in enumerate(self.envs)]

    def reset(self):
        return [env.reset() for env in self.envs]

    def step(self, actions):
        return list(zip(*[env.step(a) for env, a in zip(self.envs, actions)]))