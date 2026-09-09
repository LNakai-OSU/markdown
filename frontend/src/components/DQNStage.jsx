import { useState } from "react";
import { useArtifact } from "../lib/useArtifact";
import { useDqnModel } from "../lib/useDqnModel";
import PolicyHeatmap from "./PolicyHeatmap";
import EpisodePlayer from "./EpisodePlayer";
import MiniLineChart, { SERIES_COLORS } from "./MiniLineChart";
import { PRICES } from "../lib/demand";

const BASELINE_LABELS = {
  random: "Random price",
  fixed_best_static_price: "Fixed best static price",
  stage2_qtable_ignores_rich_state: "Stage-2 policy (ignores market signals)",
  dqn: "DQN (sees competitor price + demand shock)",
};
const BASELINE_ORDER = ["random", "fixed_best_static_price", "stage2_qtable_ignores_rich_state", "dqn"];

export default function DQNStage() {
  const { data } = useArtifact("dqn.json");
  const { data: qData } = useArtifact("qlearning.json");
  const { ready, error, predict } = useDqnModel();

  if (!data) return <p className="empty-state md-body-medium">Loading...</p>;

  const maxMean = Math.max(...BASELINE_ORDER.map((k) => data.baselines[k].mean));

  return (
    <div>
      <h2 className="section-heading md-headline-small">Now the market talks back</h2>
      <p className="md-body-medium" style={{ color: "var(--md-sys-color-on-surface-variant)", marginBottom: "1rem", maxWidth: "72ch" }}>
        Same deadline, same inventory - plus a competitor whose price drifts day to day, and a demand "mood" that
        swings market-wide. That's two more continuous numbers in the state, which is exactly where a table stops
        being an option and a small neural network (DQN) takes over.
      </p>

      <div className="m3-card m3-card-filled" style={{ padding: "1.2rem", marginBottom: "1.5rem" }}>
        <h3 className="md-title-medium" style={{ marginBottom: "0.8rem" }}>Does the extra information actually help?</h3>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.7rem" }}>
          {BASELINE_ORDER.map((key, i) => {
            const { mean, se } = data.baselines[key];
            return (
              <div key={key}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 4 }}>
                  <span className="md-body-medium">{BASELINE_LABELS[key]}</span>
                  <span className="mono md-body-medium">${mean.toFixed(0)} <span style={{ color: "var(--md-sys-color-on-surface-variant)" }}>± {se.toFixed(0)}</span></span>
                </div>
                <div style={{ background: "var(--md-sys-color-surface-container-highest)", borderRadius: 6, height: 10 }}>
                  <div style={{ width: `${(mean / maxMean) * 100}%`, height: "100%", borderRadius: 6, background: SERIES_COLORS[i % SERIES_COLORS.length] }} />
                </div>
              </div>
            );
          })}
        </div>
        <p className="md-body-small" style={{ color: "var(--md-sys-color-on-surface-variant)", marginTop: "1rem" }}>
          The DQN beats the state-blind Stage-2 policy by actually reacting to the competitor and demand signals -
          not by a huge margin (the market conditions here are moderate on purpose), but consistently, which is the
          honest result: extra state helps exactly as much as that state actually moves the optimal price.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem", marginBottom: "1.5rem" }}>
        <div className="m3-card m3-card-elevated" style={{ padding: "1.2rem" }}>
          <h3 className="md-title-medium" style={{ marginBottom: "0.6rem" }}>Training curve (20,000 episodes)</h3>
          <MiniLineChart series={[{ name: "Total reward per episode", color: SERIES_COLORS[0], data: data.reward_history_binned }]} xLabel="training progress" yLabel="$ per episode" height={220} />
        </div>
        <div className="m3-card m3-card-elevated" style={{ padding: "1.2rem" }}>
          <h3 className="md-title-medium" style={{ marginBottom: "0.6rem" }}>TD loss</h3>
          <MiniLineChart series={[{ name: "Loss", color: SERIES_COLORS[3], data: data.loss_history_binned }]} xLabel="training progress" yLabel="loss" height={220} />
        </div>
      </div>

      <div className="m3-card m3-card-elevated" style={{ padding: "1.2rem", marginBottom: "1.5rem" }}>
        <h3 className="md-title-medium" style={{ marginBottom: "0.4rem" }}>Under calm market conditions, it rediscovers Stage 2's shape</h3>
        <p className="md-body-small" style={{ color: "var(--md-sys-color-on-surface-variant)", marginBottom: "0.8rem" }}>
          Slicing the DQN's policy at a neutral competitor price and zero demand shock should look like the Stage 2
          heatmap, even though this agent never saw a table - a sanity check that it learned the same underlying
          days/inventory logic, not something unrelated.
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
          <PolicyHeatmap grid={data.policy_slice_neutral} title="DQN, neutral market conditions" maxInventory={qData?.start_inventory || 40} />
          {qData && <PolicyHeatmap grid={qData.dp_price_grid} title="Stage 2 exact optimum (for comparison)" maxInventory={qData.start_inventory} />}
        </div>
      </div>

      <div className="m3-card m3-card-elevated" style={{ padding: "1.2rem", marginBottom: "1.5rem" }}>
        <h3 className="md-title-medium" style={{ marginBottom: "0.6rem" }}>What it actually reacts to</h3>
        <p className="md-body-small" style={{ color: "var(--md-sys-color-on-surface-variant)", marginBottom: "1rem" }}>
          Fixing days-remaining and inventory, and varying only one market signal at a time.
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1.2rem" }}>
          {data.sensitivity.map((row) => (
            <div key={`${row.days}-${row.inventory}`}>
              <p className="md-label-medium" style={{ marginBottom: "0.4rem" }}>{row.days}d left, {row.inventory} units</p>
              <SensitivityMini label="vs. competitor price" points={row.by_competitor_price} xKey="competitor_price" />
              <SensitivityMini label="vs. demand shock" points={row.by_demand_shock} xKey="demand_shock" />
            </div>
          ))}
        </div>
      </div>

      <div className="m3-card m3-card-elevated" style={{ padding: "1.2rem" }}>
        <h3 className="md-title-medium" style={{ marginBottom: "0.6rem" }}>Watch it play, live</h3>
        <p className="md-body-small" style={{ color: "var(--md-sys-color-on-surface-variant)", marginBottom: "1rem" }}>
          The actual trained network, running inference in your browser via ONNX Runtime Web - no server call per
          step - against the state-blind Stage-2 policy, on the same rich-market environment.
        </p>
        {error && <p className="md-body-small" style={{ color: "var(--md-sys-color-error)" }}>Couldn't load the model: {error}</p>}
        {!ready && !error && <p className="empty-state md-body-medium">Loading the trained network...</p>}
        {ready && qData && (
          <EpisodePlayer
            richState={true}
            stepMs={220}
            policies={[
              { name: "DQN", color: SERIES_COLORS[0], policyFn: (s) => predict(s) },
              { name: "Stage-2 (state-blind)", color: SERIES_COLORS[2], policyFn: ([days, inv]) => PRICES.indexOf(qData.dp_price_grid[days][inv]) },
            ]}
          />
        )}
      </div>
    </div>
  );
}

function SensitivityMini({ label, points, xKey }) {
  return (
    <div style={{ marginBottom: "0.8rem" }}>
      <p className="md-label-small" style={{ color: "var(--md-sys-color-on-surface-variant)", marginBottom: 4 }}>{label}</p>
      <div style={{ display: "flex", gap: 4 }}>
        {points.map((p) => (
          <div key={p[xKey]} style={{ flex: 1, textAlign: "center" }}>
            <div className="mono md-label-small" style={{ color: "var(--md-sys-color-primary)" }}>${p.price}</div>
            <div className="md-label-small" style={{ color: "var(--md-sys-color-on-surface-variant)" }}>{p[xKey]}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
