"""
Stage 2: tabular Q-learning on the (days_remaining, inventory) state.

Off-policy TD control - the classic Q-learning update:
    Q(s,a) <- Q(s,a) + lr * (r + gamma * max_a' Q(s',a') - Q(s,a))
with epsilon-greedy exploration decayed over training. No neural net, no
function approximation - every (state, action) pair gets its own entry in
a table, which is only possible because this stage's state space is small.
"""

import numpy as np

from env.demand import PRICES
from env.revenue_env import N_DAYS, START_INVENTORY, RevenueManagementEnv


class QLearningAgent:
    def __init__(self, gamma=0.995, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay_episodes=8000,
                 lr_min=0.01, visit_decay_power=0.6):
        self.n_actions = len(PRICES)
        self.Q = np.zeros((N_DAYS + 1, START_INVENTORY + 1, self.n_actions))
        # Visit counts per (state, action) drive a decaying per-cell learning
        # rate (~1/visits^power) - the standard fix for tabular Q-learning
        # convergence. A single constant lr either updates rarely-visited
        # cells too slowly or lets frequently-visited ones bounce around
        # forever; this lets each cell settle at its own pace.
        self.visits = np.zeros((N_DAYS + 1, START_INVENTORY + 1, self.n_actions), dtype=np.int64)
        self.lr_min = lr_min
        self.visit_decay_power = visit_decay_power
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes

    def epsilon(self, episode):
        t = min(1.0, episode / self.epsilon_decay_episodes)
        return self.epsilon_start + t * (self.epsilon_end - self.epsilon_start)

    def act(self, state, episode, rng, greedy=False):
        days, inv = state
        if not greedy and rng.random() < self.epsilon(episode):
            return rng.integers(0, self.n_actions)
        return int(np.argmax(self.Q[days, inv]))

    def update(self, state, action, reward, next_state, done):
        days, inv = state
        target = reward
        if not done:
            ndays, ninv = next_state
            target += self.gamma * np.max(self.Q[ndays, ninv])
        self.visits[days, inv, action] += 1
        lr = max(self.lr_min, 1.0 / (self.visits[days, inv, action] ** self.visit_decay_power))
        td_error = target - self.Q[days, inv, action]
        self.Q[days, inv, action] += lr * td_error

    def greedy_policy_grid(self):
        return np.argmax(self.Q, axis=2)


def train(n_episodes=80000, seed=0, checkpoint_episodes=None):
    """checkpoint_episodes: episode counts at which to snapshot the greedy
    policy grid (a deep copy of Q at that point), so training progress can
    be scrubbed through afterward rather than only seeing the end state."""
    rng = np.random.default_rng(seed)
    env = RevenueManagementEnv(rich_state=False, rng=rng)
    agent = QLearningAgent(epsilon_decay_episodes=int(n_episodes * 0.8))

    checkpoint_set = set(checkpoint_episodes or [])
    checkpoints = {}

    reward_history = []
    for ep in range(n_episodes):
        if ep in checkpoint_set:
            checkpoints[ep] = agent.Q.copy()
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = agent.act(state, ep, rng)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        reward_history.append(total_reward)

    if n_episodes in checkpoint_set:
        checkpoints[n_episodes] = agent.Q.copy()

    return agent, reward_history, checkpoints
