import common

import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import torch.nn as nn
import torch as T 
import numpy as np
import ptan
import gym

from torch.utils.tensorboard import SummaryWriter

GAMMA = 0.99 
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
NUM_ENVS = 50
REWARD_STEP = 4
CLIP_GRAD = 0.1


class AtariA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariA2C, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_output_size = self. _get_conv_out(input_shape)

        self.policy = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
    
        self.value = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(T.zeros(1, *shape))
        return int(np.prod(o.shape))
    
    def forward(self, x):
        x = x.float() / 256
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)

    
def unpack_batch(batch, net, device='cpu'):

    states, actions, rewards = [], [], []
    not_done_idx = []
    last_states = []

    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))

    states_v = T.FloatTensor(states).to(device)
    actions_t = T.LongTensor(actions).to(device)
    rewards_np = np.array(rewards, dtype=np.float32)

    if not_done_idx:
        last_states_v = T.FloatTensor(last_states).to(device)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += GAMMA ** REWARD_STEP * last_vals_np
    ref_vals_v = T.FloatTensor(rewards_np).to(device)
    
    return states_v, actions_t, ref_vals_v

if __name__ == "__main__":
    device = T.device("cuda" if T.cuda.is_available() else "cpu")

    make_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make("BreakoutNoFrameskip-v4"))
    envs = [make_env() for _ in range(NUM_ENVS)]
    writer = SummaryWriter("run")

    net = AtariA2C(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    print(net)

    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEP)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    batch = []

    with common.RewardTracker(writer, stop_reward=500) as tracker:
        with ptan.common.utils.TBMeanTracker(batch, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                batch.append(exp)
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards:
                    if tracker.reward(new_rewards[0], step_idx):
                        break
                
                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_t, vals_ref_v = unpack_batch(batch, net, device=device)
                batch.clear()
                
                optimizer.zero_grad()
                logits_v, values_v = net(states_v)
                loss_value_v = F.mse_loss(values_v.squeeze(-1), vals_ref_v)
                
                log_prob_v = F.log_softmax(logits_v, dim=1)
                adv_v = vals_ref_v - values_v.detach()
                log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
                loss_policy_v = -log_prob_actions_v.mean()
                
                loss_policy_v.backward(retain_graph=True)

                prob_v = F.softmax(logits_v, dim=1)
                entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

                loss_v = entropy_loss_v + loss_value_v
                loss_v.backward()
                optimizer.step()
                   
                   



                
                