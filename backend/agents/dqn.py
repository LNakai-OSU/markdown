"""
Stage 3: Deep Q-Network on the rich-state environment (competitor price +
demand shock added on top of days/inventory - see revenue_env.py for why
that makes the state space impractical to tabulate).

Standard DQN components, nothing exotic: a small MLP as the Q-function
approximator, a replay buffer to decorrelate consecutive transitions, and a
periodically-synced target network to keep the TD target from chasing a
network that's changing under it every step.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env.demand import PRICES
from env.revenue_env import N_DAYS, START_INVENTORY, RevenueManagementEnv
from env.demand import PRICE_MIN, PRICE_MAX

STATE_DIM = 4  # days_remaining, inventory, competitor_price, demand_shock


def normalize_state(state):
    days, inv, comp_price, shock = state
    return np.array([
        days / N_DAYS,
        inv / START_INVENTORY,
        (comp_price - PRICE_MIN) / (PRICE_MAX - PRICE_MIN),
        np.clip(shock, -2, 2) / 2,
    ], dtype=np.float32)


class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, s, a, r, ns, done):
        item = (s, a, r, ns, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, rng):
        idx = rng.integers(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in idx]
        s, a, r, ns, done = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a), np.array(r, dtype=np.float32),
                np.array(ns, dtype=np.float32), np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, n_actions=len(PRICES), lr=1e-3, gamma=0.995,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=30000,
                 target_sync_every=500):
        self.n_actions = n_actions
        self.gamma = gamma
        self.q_net = QNetwork(STATE_DIM, n_actions)
        self.target_net = QNetwork(STATE_DIM, n_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_sync_every = target_sync_every
        self.step_count = 0

    def epsilon(self):
        t = min(1.0, self.step_count / self.epsilon_decay_steps)
        return self.epsilon_start + t * (self.epsilon_end - self.epsilon_start)

    def act(self, state_norm, rng, greedy=False):
        if not greedy and rng.random() < self.epsilon():
            return rng.integers(0, self.n_actions)
        with torch.no_grad():
            q = self.q_net(torch.from_numpy(state_norm).unsqueeze(0))
            return int(torch.argmax(q, dim=1).item())

    def train_step(self, batch_size, rng):
        if len(self.buffer) < batch_size:
            return None
        s, a, r, ns, done = self.buffer.sample(batch_size, rng)
        s_t = torch.from_numpy(s)
        ns_t = torch.from_numpy(ns)
        a_t = torch.from_numpy(a).long().unsqueeze(1)
        r_t = torch.from_numpy(r)
        done_t = torch.from_numpy(done)

        q_values = self.q_net(s_t).gather(1, a_t).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(ns_t).max(dim=1).values
            target = r_t + self.gamma * next_q * (1 - done_t)

        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_sync_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return loss.item()


def train(n_episodes=20000, batch_size=64, seed=0, lr=5e-4):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = RevenueManagementEnv(rich_state=True, rng=rng)
    agent = DQNAgent(epsilon_decay_steps=n_episodes * 15, lr=lr)

    reward_history = []
    loss_history = []
    for ep in range(n_episodes):
        state = env.reset()
        state_norm = normalize_state(state)
        total_reward = 0.0
        done = False
        while not done:
            action = agent.act(state_norm, rng)
            next_state, reward, done, _ = env.step(action)
            next_state_norm = normalize_state(next_state)
            agent.buffer.push(state_norm, action, reward, next_state_norm, done)
            loss = agent.train_step(batch_size, rng)
            if loss is not None:
                loss_history.append(loss)
            state_norm = next_state_norm
            total_reward += reward
        reward_history.append(total_reward)

    return agent, reward_history, loss_history
