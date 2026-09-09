// JS port of backend/agents/bandits.py, for a live in-browser run (one
// seed, watched in real time) that sits alongside the smoothed
// many-seed regret curves computed offline in Python.

import { PRICES } from "./demand.js";

function gaussian() {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// Marsaglia-Tsang, valid for shape >= 1 (always true here: the Gamma-Poisson
// posterior's alpha starts at the prior's 1.0 and only grows).
function gammaSample(shape) {
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do {
      x = gaussian();
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = Math.random();
    if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) {
      return d * v;
    }
  }
}

export class EpsilonGreedyBandit {
  static label = "Epsilon-Greedy";
  constructor(nArms, epsilonStart = 1.0, epsilonEnd = 0.02, decaySteps = 2000) {
    this.counts = new Array(nArms).fill(0);
    this.sums = new Array(nArms).fill(0);
    this.epsilonStart = epsilonStart;
    this.epsilonEnd = epsilonEnd;
    this.decaySteps = decaySteps;
    this.t = 0;
  }
  epsilon() {
    const frac = Math.min(1, this.t / this.decaySteps);
    return this.epsilonStart + frac * (this.epsilonEnd - this.epsilonStart);
  }
  selectArm() {
    if (Math.random() < this.epsilon() || this.t < this.counts.length) {
      return Math.floor(Math.random() * this.counts.length);
    }
    let best = 0, bestMean = -Infinity;
    for (let i = 0; i < this.counts.length; i++) {
      const mean = this.counts[i] > 0 ? this.sums[i] / this.counts[i] : 0;
      if (mean > bestMean) { bestMean = mean; best = i; }
    }
    return best;
  }
  update(arm, reward) {
    this.counts[arm] += 1;
    this.sums[arm] += reward;
    this.t += 1;
  }
  estimates() {
    return this.counts.map((c, i) => (c > 0 ? this.sums[i] / c : 0));
  }
}

export class UCB1Bandit {
  static label = "UCB1";
  constructor(nArms, c = 40.0) {
    this.counts = new Array(nArms).fill(0);
    this.sums = new Array(nArms).fill(0);
    this.c = c;
    this.t = 0;
  }
  selectArm() {
    if (this.t < this.counts.length) return this.t;
    let best = 0, bestScore = -Infinity;
    for (let i = 0; i < this.counts.length; i++) {
      const mean = this.sums[i] / this.counts[i];
      const bonus = this.c * Math.sqrt(Math.log(this.t + 1) / this.counts[i]);
      const score = mean + bonus;
      if (score > bestScore) { bestScore = score; best = i; }
    }
    return best;
  }
  update(arm, reward) {
    this.counts[arm] += 1;
    this.sums[arm] += reward;
    this.t += 1;
  }
  estimates() {
    return this.counts.map((c, i) => (c > 0 ? this.sums[i] / c : 0));
  }
}

export class ThompsonSamplingBandit {
  static label = "Thompson Sampling";
  constructor(nArms, priorAlpha = 1.0, priorBeta = 1.0) {
    this.alpha = new Array(nArms).fill(priorAlpha);
    this.beta = new Array(nArms).fill(priorBeta);
  }
  selectArm() {
    let best = 0, bestVal = -Infinity;
    for (let i = 0; i < this.alpha.length; i++) {
      const lambdaSample = gammaSample(this.alpha[i]) / this.beta[i];
      const val = PRICES[i] * lambdaSample;
      if (val > bestVal) { bestVal = val; best = i; }
    }
    return best;
  }
  update(arm, reward) {
    const price = PRICES[arm];
    const unitsSold = price > 0 ? reward / price : 0;
    this.alpha[arm] += unitsSold;
    this.beta[arm] += 1;
  }
  estimates() {
    return this.alpha.map((a, i) => PRICES[i] * (a / this.beta[i]));
  }
}

export const BANDIT_ALGORITHMS = {
  epsilon_greedy: EpsilonGreedyBandit,
  ucb1: UCB1Bandit,
  thompson: ThompsonSamplingBandit,
};
