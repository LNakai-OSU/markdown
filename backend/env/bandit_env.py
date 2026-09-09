"""
Stage 1 environment: a stateless pricing bandit.

Deliberately unrealistic in one specific way - infinite restock, no
deadline - to isolate the exploration/exploitation problem before Stage 2
adds the inventory + deadline state that turns this into a real MDP. Each
"pull" of price-arm p yields a fresh Poisson(rate(p)) number of units sold
that round, so the reward distribution per arm is stationary: the agent is
purely trying to find the revenue-maximizing price, nothing more.
"""

import numpy as np

from .demand import PRICES, base_rate


class PricingBanditEnv:
    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()
        self.n_arms = len(PRICES)

    def pull(self, arm_idx):
        price = PRICES[arm_idx]
        units_sold = self.rng.poisson(base_rate(price))
        reward = price * units_sold
        return reward
