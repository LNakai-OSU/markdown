"""
Stage 1: three classic bandit algorithms, all solving the same stateless
"which price maximizes revenue" problem, so their regret curves can be
compared directly. This is the actual vocabulary of real online
experimentation/personalization systems, not just a toy.

Each arm's reward per pull is price * Poisson(rate(price)) - unbounded,
not the [0,1] rewards most textbook bandit writeups assume, so the
exploration terms below are tuned for that scale rather than lifted
verbatim from a textbook.
"""

import numpy as np

from env.demand import PRICES


class EpsilonGreedyBandit:
    """Sample-average reward estimate per arm; epsilon decays over time so
    exploration is heavy early and the agent settles down later."""

    name = "Epsilon-Greedy"

    def __init__(self, n_arms, epsilon_start=1.0, epsilon_end=0.02, decay_steps=2000):
        self.counts = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_steps = decay_steps
        self.t = 0

    def epsilon(self):
        frac = min(1.0, self.t / self.decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_arm(self, rng):
        if rng.random() < self.epsilon() or self.t < len(self.counts):
            return rng.integers(0, len(self.counts))
        means = np.where(self.counts > 0, self.sums / np.maximum(self.counts, 1), 0)
        return int(np.argmax(means))

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.sums[arm] += reward
        self.t += 1


class UCB1Bandit:
    """Optimism under uncertainty: pick the arm with the highest
    mean + exploration bonus, where the bonus shrinks as an arm gets
    pulled more. `c` is scaled to this problem's actual reward magnitude
    (tens to low hundreds), not the unit scale UCB1 is usually written for."""

    name = "UCB1"

    def __init__(self, n_arms, c=40.0):
        self.counts = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)
        self.c = c
        self.t = 0

    def select_arm(self, rng):
        if self.t < len(self.counts):
            return self.t  # pull every arm once first
        means = self.sums / self.counts
        bonus = self.c * np.sqrt(np.log(self.t + 1) / self.counts)
        return int(np.argmax(means + bonus))

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.sums[arm] += reward
        self.t += 1


class ThompsonSamplingBandit:
    """Bayesian approach: model each arm's demand rate as Poisson(lambda),
    put a Gamma(1,1) conjugate prior on lambda, and sample from the
    posterior each round rather than using a point estimate. Naturally
    balances exploration and exploitation through posterior uncertainty
    instead of a hand-tuned bonus term."""

    name = "Thompson Sampling"

    def __init__(self, n_arms, prior_alpha=1.0, prior_beta=1.0):
        self.alpha = np.full(n_arms, prior_alpha)
        self.beta = np.full(n_arms, prior_beta)

    def select_arm(self, rng):
        samples = rng.gamma(self.alpha, 1.0 / self.beta)
        expected_revenue = np.array(PRICES) * samples
        return int(np.argmax(expected_revenue))

    def update(self, arm, reward):
        price = PRICES[arm]
        units_sold = reward / price if price > 0 else 0
        self.alpha[arm] += units_sold
        self.beta[arm] += 1
