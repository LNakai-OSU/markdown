"""
Runs all three training stages once and dumps everything the frontend
needs as static JSON (+ one ONNX model) into frontend/public/data/. No
backend runs at request time - training happens here, offline, and the
trained artifacts ship as static files the browser reads and, for the
Q-table and DQN, actually executes itself (see the frontend's env.js for
the ported simulation and onnxruntime-web for the DQN forward pass).
"""

import json
import os
import time

import numpy as np
import torch

from env.demand import PRICES, expected_revenue, best_static_price
from env.bandit_env import PricingBanditEnv
from env.revenue_env import RevenueManagementEnv, N_DAYS, START_INVENTORY
from agents.bandits import EpsilonGreedyBandit, UCB1Bandit, ThompsonSamplingBandit
from agents.qlearning import train as train_qlearning
from agents.dqn import train as train_dqn, normalize_state
from dp_solver import solve as dp_solve

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "public", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def save_json(name, obj):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f)
    print(f"wrote {path} ({os.path.getsize(path) / 1024:.1f} KB)")


def rollout(env_factory, policy_fn, n_episodes=2000, seed=999):
    rng = np.random.default_rng(seed)
    env = env_factory(rng)
    totals = []
    for _ in range(n_episodes):
        s = env.reset()
        done = False
        total = 0.0
        while not done:
            a = policy_fn(s)
            s, r, done, _ = env.step(a)
            total += r
        totals.append(total)
    return float(np.mean(totals)), float(np.std(totals) / np.sqrt(len(totals)))


def bin_series(series, n_bins=150):
    series = np.array(series, dtype=float)
    if len(series) <= n_bins:
        return series.tolist()
    edges = np.linspace(0, len(series), n_bins + 1).astype(int)
    return [float(np.mean(series[edges[i]:edges[i + 1]])) for i in range(n_bins)]


# ---------------------------------------------------------------- Stage 1 --
def export_bandit_stage():
    print("\n=== Stage 1: bandits ===")
    best_price = best_static_price()
    best_rev = expected_revenue(best_price)
    n_rounds = 2500
    n_seeds = 40

    algos = {
        "epsilon_greedy": EpsilonGreedyBandit,
        "ucb1": UCB1Bandit,
        "thompson": ThompsonSamplingBandit,
    }
    results = {}
    for key, AgentCls in algos.items():
        regret_curves = np.zeros((n_seeds, n_rounds))
        final_estimates = None
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            env = PricingBanditEnv(rng=rng)
            agent = AgentCls(len(PRICES))
            cum = 0.0
            for t in range(n_rounds):
                arm = agent.select_arm(rng)
                reward = env.pull(arm)
                agent.update(arm, reward)
                cum += best_rev - expected_revenue(PRICES[arm])
                regret_curves[seed, t] = cum
            if seed == 0:
                if hasattr(agent, "sums"):
                    counts = np.maximum(agent.counts, 1)
                    final_estimates = (agent.sums / counts).tolist()
                else:
                    final_estimates = (np.array(PRICES) * (agent.alpha / agent.beta)).tolist()
        avg_curve = regret_curves.mean(axis=0)
        results[key] = {
            "name": AgentCls.name,
            "regret_curve": bin_series(avg_curve, 150),
            "final_estimates": final_estimates,
        }
        print(f"{AgentCls.name:20s} final avg cumulative regret: {avg_curve[-1]:.0f}")

    save_json("bandit.json", {
        "prices": PRICES,
        "best_price": best_price,
        "best_expected_revenue": best_rev,
        "expected_revenue_by_price": [expected_revenue(p) for p in PRICES],
        "n_rounds": n_rounds,
        "algorithms": results,
    })


# ---------------------------------------------------------------- Stage 2 --
def export_qlearning_stage():
    print("\n=== Stage 2: tabular Q-learning ===")
    t0 = time.time()
    n_episodes = 80000
    checkpoint_eps = [0, 500, 2000, 8000, 20000, 80000]
    agent, history, checkpoints = train_qlearning(n_episodes=n_episodes, seed=0, checkpoint_episodes=checkpoint_eps)
    print(f"trained in {time.time() - t0:.1f}s")

    V, dp_policy = dp_solve()
    dp_price_grid = [[PRICES[int(dp_policy[d, i])] for i in range(START_INVENTORY + 1)] for d in range(N_DAYS + 1)]
    dp_value = float(V[N_DAYS, START_INVENTORY])

    def env_factory(rng):
        return RevenueManagementEnv(rich_state=False, rng=rng)

    checkpoint_data = []
    for ep in checkpoint_eps:
        Q = checkpoints[ep]
        policy_grid = np.argmax(Q, axis=2)
        price_grid = [[PRICES[int(policy_grid[d, i])] for i in range(START_INVENTORY + 1)] for d in range(N_DAYS + 1)]
        mean_r, se_r = rollout(env_factory, lambda s, Q=Q: int(np.argmax(Q[s[0], s[1]])), n_episodes=600, seed=42)
        checkpoint_data.append({"episode": ep, "price_grid": price_grid, "avg_reward": mean_r, "avg_reward_se": se_r})
        print(f"checkpoint ep={ep:6d}: avg reward {mean_r:.1f} +/- {se_r:.1f}")

    mean_dp, se_dp = rollout(env_factory, lambda s: int(dp_policy[s[0], s[1]]), n_episodes=2000, seed=42)
    print(f"DP-optimal rollout: {mean_dp:.1f} +/- {se_dp:.1f} (theoretical {dp_value:.1f})")

    save_json("qlearning.json", {
        "prices": PRICES,
        "n_days": N_DAYS,
        "start_inventory": START_INVENTORY,
        "dp_price_grid": dp_price_grid,
        "dp_value": dp_value,
        "dp_rollout_value": mean_dp,
        "reward_history_binned": bin_series(history, 200),
        "checkpoints": checkpoint_data,
    })


# ---------------------------------------------------------------- Stage 3 --
def export_dqn_stage():
    print("\n=== Stage 3: DQN ===")
    t0 = time.time()
    agent, history, losses = train_dqn(n_episodes=20000, seed=0)
    print(f"trained in {time.time() - t0:.1f}s")

    def rich_env_factory(rng):
        return RevenueManagementEnv(rich_state=True, rng=rng)

    def dqn_policy(s):
        sn = normalize_state(s)
        with torch.no_grad():
            q = agent.q_net(torch.from_numpy(sn).unsqueeze(0))
            return int(torch.argmax(q, dim=1).item())

    # Baseline: the tabular Q-learning policy trained on the SIMPLE state,
    # applied here while ignoring competitor price / demand shock entirely -
    # this is the "was the richer state worth it" comparison.
    q_agent, _, _ = train_qlearning(n_episodes=80000, seed=0)

    def q_table_policy(s):
        return int(np.argmax(q_agent.Q[s[0], s[1]]))

    best_p_idx = PRICES.index(best_static_price())
    baselines = {}
    for key, fn, seed_fn in [
        ("random", lambda s: int(np.random.default_rng(0).integers(0, len(PRICES))), None),
        ("fixed_best_static_price", lambda s: best_p_idx, None),
        ("stage2_qtable_ignores_rich_state", q_table_policy, None),
        ("dqn", dqn_policy, None),
    ]:
        mean, se = rollout(rich_env_factory, fn, n_episodes=2000, seed=42)
        baselines[key] = {"mean": mean, "se": se}
        print(f"{key:35s} {mean:.1f} +/- {se:.1f}")

    # A policy slice at "neutral" market conditions, directly comparable to
    # the Stage 2 heatmap, plus a few points showing how the recommended
    # price shifts as competitor price / demand shock move away from neutral.
    def dqn_price_at(days, inv, comp_price, shock):
        sn = normalize_state((days, inv, comp_price, shock))
        with torch.no_grad():
            q = agent.q_net(torch.from_numpy(sn).unsqueeze(0))
            return PRICES[int(torch.argmax(q, dim=1).item())]

    neutral_grid = [[dqn_price_at(d, i, 110.0, 0.0) for i in range(START_INVENTORY + 1)] for d in range(N_DAYS + 1)]

    sensitivity = []
    sample_points = [(20, 20), (10, 10), (5, 30)]
    for days, inv in sample_points:
        row = {"days": days, "inventory": inv, "by_competitor_price": [], "by_demand_shock": []}
        for cp in [60, 90, 110, 130, 160]:
            row["by_competitor_price"].append({"competitor_price": cp, "price": dqn_price_at(days, inv, cp, 0.0)})
        for shock in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            row["by_demand_shock"].append({"demand_shock": shock, "price": dqn_price_at(days, inv, 110.0, shock)})
        sensitivity.append(row)

    save_json("dqn.json", {
        "reward_history_binned": bin_series(history, 200),
        "loss_history_binned": bin_series(losses, 200),
        "baselines": baselines,
        "policy_slice_neutral": neutral_grid,
        "sensitivity": sensitivity,
    })

    dummy = torch.zeros(1, 4)
    onnx_path = os.path.join(OUT_DIR, "dqn_model.onnx")
    torch.onnx.export(
        agent.q_net, dummy, onnx_path,
        input_names=["state"], output_names=["q_values"],
        dynamic_axes={"state": {0: "batch"}, "q_values": {0: "batch"}},
        opset_version=17,
    )
    print(f"wrote {onnx_path} ({os.path.getsize(onnx_path) / 1024:.1f} KB)")


if __name__ == "__main__":
    export_bandit_stage()
    export_qlearning_stage()
    export_dqn_stage()
    print("\nAll artifacts exported.")
