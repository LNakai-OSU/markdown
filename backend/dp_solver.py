"""
Exact solution to the (days_remaining, inventory) revenue-management MDP
via backward induction (dynamic programming), independent of any learning
algorithm. This exists purely as ground truth: if tabular Q-learning
converged correctly, its learned policy should match this one. It's the
"prove it, don't just claim it" check for Stage 2.

Only tractable because the simple-stage state space is small (30 days x 40
units). The whole point of Stage 3's richer, continuous state is that this
approach stops being an option - you can't backward-induct over a
continuum, which is the actual reason DQN exists.
"""

import numpy as np
from scipy.stats import poisson

from env.demand import PRICES, base_rate
from env.revenue_env import N_DAYS, START_INVENTORY, SALVAGE_PENALTY_PER_UNIT


def solve():
    n_prices = len(PRICES)
    # V[d][i] = optimal expected future reward with d days left and i units in stock
    V = np.zeros((N_DAYS + 1, START_INVENTORY + 1))
    policy = np.zeros((N_DAYS + 1, START_INVENTORY + 1), dtype=int)

    V[0, :] = -SALVAGE_PENALTY_PER_UNIT * np.arange(START_INVENTORY + 1)
    V[0, 0] = 0.0

    for d in range(1, N_DAYS + 1):
        V[d, 0] = 0.0
        for i in range(1, START_INVENTORY + 1):
            best_value = -np.inf
            best_price_idx = 0
            for p_idx, price in enumerate(PRICES):
                rate = base_rate(price)
                k = np.arange(0, i)  # demand outcomes that don't exhaust stock
                pmf = poisson.pmf(k, rate)
                partial_sell = np.sum(pmf * (price * k + V[d - 1, i - k]))
                p_sellout = 1.0 - poisson.cdf(i - 1, rate)
                sellout_value = p_sellout * (price * i + V[d - 1, 0])
                q = partial_sell + sellout_value
                if q > best_value:
                    best_value = q
                    best_price_idx = p_idx
            V[d, i] = best_value
            policy[d, i] = best_price_idx

    return V, policy


if __name__ == "__main__":
    V, policy = solve()
    print("Optimal expected revenue from (30 days, 40 units):", round(V[N_DAYS, START_INVENTORY], 2))
    print("\nOptimal price policy (rows=days remaining, cols=inventory), showing every 5th day/unit:")
    for d in range(N_DAYS, -1, -5):
        row = [PRICES[policy[d, i]] for i in range(0, START_INVENTORY + 1, 5)]
        print(f"day {d:2d}: {row}")
