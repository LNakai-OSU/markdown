// JS port of backend/env/revenue_env.py, for live client-side episode
// playback (Stage 2's tabular policy and Stage 3's DQN both get stepped
// through this in the browser - no server round-trip per frame).

import { PRICES, PRICE_MIN, PRICE_MAX, baseRate, poissonSample } from "./demand.js";

export const N_DAYS = 30;
export const START_INVENTORY = 40;
export const SALVAGE_PENALTY_PER_UNIT = 15;

export class RevenueManagementEnv {
  constructor(richState = false) {
    this.richState = richState;
    this.reset();
  }

  reset() {
    this.daysRemaining = N_DAYS;
    this.inventory = START_INVENTORY;
    this.competitorPrice = 110.0;
    this.demandShock = 0.0;
    this.done = false;
    return this.state();
  }

  state() {
    if (this.richState) {
      return [this.daysRemaining, this.inventory, this.competitorPrice, this.demandShock];
    }
    return [this.daysRemaining, this.inventory];
  }

  step(actionIdx) {
    if (this.done) throw new Error("step() called on a finished episode");

    const price = PRICES[actionIdx];
    let rate = baseRate(price);

    if (this.richState) {
      const competitorFactor = 1.0 + 0.3 * (this.competitorPrice - price) / (PRICE_MAX - PRICE_MIN);
      rate = rate * Math.max(0.05, competitorFactor) * Math.exp(this.demandShock);
    }

    const demand = poissonSample(Math.max(rate, 0));
    const unitsSold = Math.min(demand, this.inventory);
    const revenue = price * unitsSold;

    this.inventory -= unitsSold;
    this.daysRemaining -= 1;

    if (this.richState) {
      const gaussian = () => {
        // Box-Muller
        const u1 = Math.random(), u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      };
      this.competitorPrice = Math.min(PRICE_MAX, Math.max(PRICE_MIN, this.competitorPrice + gaussian() * 5));
      this.demandShock = 0.9 * this.demandShock + gaussian() * 0.3;
    }

    this.done = this.daysRemaining <= 0 || this.inventory <= 0;
    let reward = revenue;
    if (this.done && this.inventory > 0) {
      reward -= SALVAGE_PENALTY_PER_UNIT * this.inventory;
    }

    return { state: this.state(), reward, done: this.done, price, unitsSold };
  }
}

export function normalizeState([days, inv, compPrice, shock]) {
  return Float32Array.from([
    days / N_DAYS,
    inv / START_INVENTORY,
    (compPrice - PRICE_MIN) / (PRICE_MAX - PRICE_MIN),
    Math.max(-2, Math.min(2, shock)) / 2,
  ]);
}
