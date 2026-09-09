"""
Shared demand model used by every stage (bandit, tabular Q-learning, DQN) so
the three agents are all learning about the *same* underlying market - only
what they're allowed to see and remember changes between stages.

Demand at a given price is Poisson-distributed with a rate that decays
exponentially in price (a standard textbook demand-curve shape): cheap
prices draw a lot of buyers, expensive ones draw few. Expected per-round
revenue is price * rate(price), which is a hump-shaped curve with an
interior optimum - not "always price at the minimum" or "always price at
the maximum," so an agent actually has something nontrivial to find.
"""

import numpy as np

PRICE_MIN = 20
PRICE_MAX = 200
PRICE_STEP = 20
PRICES = list(range(PRICE_MIN, PRICE_MAX + 1, PRICE_STEP))  # 10 discrete prices

LAMBDA_MAX = 5.0   # expected units demanded per period at the minimum price
DECAY_ALPHA = 3.0  # how fast demand falls off as price rises


def base_rate(price):
    """Expected units demanded per period at `price`, ignoring any
    competitor/market-condition effects (those are added on top for the
    richer DQN-stage environment)."""
    frac = (price - PRICE_MIN) / (PRICE_MAX - PRICE_MIN)
    return LAMBDA_MAX * np.exp(-DECAY_ALPHA * frac)


def expected_revenue(price):
    return price * base_rate(price)


def best_static_price():
    """The revenue-maximizing single fixed price, ignoring inventory/deadline -
    this is the target a stateless bandit is trying to find."""
    revenues = [expected_revenue(p) for p in PRICES]
    return PRICES[int(np.argmax(revenues))]
