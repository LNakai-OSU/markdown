"""
Stage 2 & 3 environment: dynamic pricing with finite inventory and a
selling deadline - the real "revenue management" problem (this is
literally how airlines price seats and fashion retailers price
markdowns). Adding this state is what turns the problem from a bandit into
a genuine MDP: the right price now depends on how much stock is left and
how many days remain, not just the static demand curve.

`rich_state=True` (used for the DQN stage) adds two continuous state
variables - a competitor's price (a random walk) and a market-condition
"demand shock" (a mean-reverting process) - that make the state space
mixed continuous/discrete and too large to discretize into a table without
either throwing away information or blowing up the table size. That's the
actual, honest reason to move to a function approximator (DQN) instead of
tabular Q-learning; it isn't just "because DQN is fancier."
"""

import numpy as np

from .demand import PRICES, PRICE_MIN, PRICE_MAX, base_rate

N_DAYS = 30
START_INVENTORY = 40
SALVAGE_PENALTY_PER_UNIT = 15  # cost of unsold stock at the deadline


class RevenueManagementEnv:
    def __init__(self, rich_state=False, rng=None):
        self.rich_state = rich_state
        self.rng = rng or np.random.default_rng()
        self.n_actions = len(PRICES)
        self.reset()

    def reset(self):
        self.days_remaining = N_DAYS
        self.inventory = START_INVENTORY
        self.competitor_price = 110.0
        self.demand_shock = 0.0
        self.done = False
        return self.state()

    def state(self):
        base = (self.days_remaining, self.inventory)
        if self.rich_state:
            return base + (self.competitor_price, self.demand_shock)
        return base

    def step(self, action_idx):
        if self.done:
            raise RuntimeError("step() called on a finished episode - call reset()")

        price = PRICES[action_idx]
        rate = base_rate(price)

        if self.rich_state:
            # Cheaper than the competitor -> demand boost; pricier -> demand hit.
            competitor_factor = 1.0 + 0.3 * (self.competitor_price - price) / (PRICE_MAX - PRICE_MIN)
            rate = rate * max(0.05, competitor_factor) * np.exp(self.demand_shock)

        demand = self.rng.poisson(max(rate, 0.0))
        units_sold = min(demand, self.inventory)
        revenue = price * units_sold

        self.inventory -= units_sold
        self.days_remaining -= 1

        if self.rich_state:
            self.competitor_price = float(np.clip(self.competitor_price + self.rng.normal(0, 5), PRICE_MIN, PRICE_MAX))
            self.demand_shock = 0.9 * self.demand_shock + self.rng.normal(0, 0.3)

        self.done = self.days_remaining <= 0 or self.inventory <= 0
        reward = revenue
        if self.done and self.inventory > 0:
            reward -= SALVAGE_PENALTY_PER_UNIT * self.inventory

        return self.state(), reward, self.done, {"units_sold": units_sold, "price": price}
