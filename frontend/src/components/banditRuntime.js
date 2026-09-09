// Thin wrapper matching backend/env/bandit_env.py's pull() for the live
// in-browser bandit race - stateless, infinite restock, no deadline.
import { PRICES, baseRate, poissonSample } from "../lib/demand";

export { PRICES };

export const PricingBanditEnvLike = {
  pull(armIdx) {
    const price = PRICES[armIdx];
    const unitsSold = poissonSample(baseRate(price));
    return price * unitsSold;
  },
};
