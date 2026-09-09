import { useState } from "react";
import { useArtifact } from "../lib/useArtifact";
import PolicyHeatmap from "./PolicyHeatmap";
import EpisodePlayer from "./EpisodePlayer";
import MiniLineChart, { SERIES_COLORS } from "./MiniLineChart";
import { PRICES } from "../lib/demand";

export default function QLearningStage() {
  const { data } = useArtifact("qlearning.json");
  const [checkpointIdx, setCheckpointIdx] = useState(0);

  if (!data) return <p className="empty-state md-body-medium">Loading...</p>;

  const checkpoint = data.checkpoints[checkpointIdx];
  const pctOfOptimal = (checkpoint.avg_reward / data.dp_rollout_value) * 100;

  function policyFromGrid(grid) {
    return ([days, inv]) => PRICES.indexOf(grid[days][inv]);
  }

  return (
    <div>
      <h2 className="section-heading md-headline-small">Now it has to last the season</h2>
      <p className="md-body-medium" style={{ color: "var(--md-sys-color-on-surface-variant)", marginBottom: "1rem", maxWidth: "72ch" }}>
        {data.start_inventory} units, {data.n_days} days, no restocking. The right price now depends on the state -
        how much is left, how many days remain - which a bandit has no way to represent. Tabular Q-learning gets a
        table of (days, inventory) &rarr; price and learns it purely from trial and error.
      </p>

      <div className="m3-card m3-card-filled" style={{ padding: "1.2rem", marginBottom: "1.5rem" }}>
        <p className="md-label-medium" style={{ color: "var(--md-sys-color-on-surface-variant)", marginBottom: "0.6rem" }}>
          Training checkpoint - drag through episodes to watch the policy sharpen
        </p>
        <div className="m3-segmented" style={{ marginBottom: "1rem", flexWrap: "wrap" }}>
          {data.checkpoints.map((c, i) => (
            <button
              key={c.episode}
              className={`m3-segment ${i === checkpointIdx ? "is-selected" : ""}`}
              onClick={() => setCheckpointIdx(i)}
            >
              {c.episode.toLocaleString()} ep
            </button>
          ))}
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
          <div>
            <PolicyHeatmap grid={checkpoint.price_grid} title="Learned policy" maxInventory={data.start_inventory} />
          </div>
          <div>
            <PolicyHeatmap grid={data.dp_price_grid} title="Exact optimal (dynamic programming)" maxInventory={data.start_inventory} />
          </div>
        </div>

        <div style={{ display: "flex", gap: "2rem", marginTop: "1rem" }}>
          <Stat label="Avg. revenue at this checkpoint" value={`$${checkpoint.avg_reward.toFixed(0)}`} sub={`± ${checkpoint.avg_reward_se.toFixed(0)}`} />
          <Stat label="Exact optimal" value={`$${data.dp_rollout_value.toFixed(0)}`} />
          <Stat label="% of optimal" value={`${pctOfOptimal.toFixed(1)}%`} highlight />
        </div>
      </div>

      <div className="m3-card m3-card-elevated" style={{ padding: "1.2rem", marginBottom: "1.5rem" }}>
        <h3 className="md-title-medium" style={{ marginBottom: "0.6rem" }}>Training curve (80,000 episodes)</h3>
        <MiniLineChart
          series={[{ name: "Total reward per episode", color: SERIES_COLORS[0], data: data.reward_history_binned }]}
          xLabel="training progress (binned)"
          yLabel="$ per episode"
          height={240}
        />
      </div>

      <div className="m3-card m3-card-elevated" style={{ padding: "1.2rem" }}>
        <h3 className="md-title-medium" style={{ marginBottom: "0.6rem" }}>Watch it play, live</h3>
        <p className="md-body-small" style={{ color: "var(--md-sys-color-on-surface-variant)", marginBottom: "1rem" }}>
          The selected checkpoint's policy vs. the exact optimum, each playing a fresh randomly-drawn selling season.
        </p>
        <EpisodePlayer
          richState={false}
          policies={[
            { name: `Checkpoint (${checkpoint.episode.toLocaleString()} ep)`, color: SERIES_COLORS[0], policyFn: policyFromGrid(checkpoint.price_grid) },
            { name: "Exact optimal", color: SERIES_COLORS[2], policyFn: policyFromGrid(data.dp_price_grid) },
          ]}
        />
      </div>
    </div>
  );
}

function Stat({ label, value, sub, highlight }) {
  return (
    <div>
      <p className="md-label-medium" style={{ color: "var(--md-sys-color-on-surface-variant)" }}>{label}</p>
      <p className="md-headline-small mono" style={{ color: highlight ? "var(--md-sys-color-primary)" : "var(--md-sys-color-on-surface)" }}>
        {value}{sub && <span className="md-body-small" style={{ marginLeft: 4, color: "var(--md-sys-color-on-surface-variant)" }}>{sub}</span>}
      </p>
    </div>
  );
}
