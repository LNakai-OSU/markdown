// JS port of backend/env/demand.py - kept numerically identical on purpose
// so a live in-browser episode is playing out the same market the Python
// training used, not an approximation of it.

export const PRICE_MIN = 20;
export const PRICE_MAX = 200;
export const PRICE_STEP = 20;
export const PRICES = [];
for (let p = PRICE_MIN; p <= PRICE_MAX; p += PRICE_STEP) PRICES.push(p);

export const LAMBDA_MAX = 5.0;
export const DECAY_ALPHA = 3.0;

export function baseRate(price) {
  const frac = (price - PRICE_MIN) / (PRICE_MAX - PRICE_MIN);
  return LAMBDA_MAX * Math.exp(-DECAY_ALPHA * frac);
}

export function expectedRevenue(price) {
  return price * baseRate(price);
}

export function bestStaticPrice() {
  let best = PRICES[0];
  let bestRev = -Infinity;
  for (const p of PRICES) {
    const rev = expectedRevenue(p);
    if (rev > bestRev) {
      bestRev = rev;
      best = p;
    }
  }
  return best;
}

// Knuth's algorithm - fine for the small rates (<10) this project uses.
export function poissonSample(lambda) {
  if (lambda <= 0) return 0;
  const L = Math.exp(-lambda);
  let k = 0;
  let p = 1;
  do {
    k += 1;
    p *= Math.random();
  } while (p > L);
  return k - 1;
}
