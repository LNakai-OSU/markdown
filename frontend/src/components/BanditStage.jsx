import { useEffect, useRef, useState } from "react";
import { useArtifact } from "../lib/useArtifact";
import { PricingBanditEnvLike, PRICES } from "./banditRuntime";
import { BANDIT_ALGORITHMS } from "../lib/bandits";
import { expectedRevenue } from "../lib/demand";
import MiniLineChart, { SERIES_COLORS } from "./MiniLineChart";

const ALGO_KEYS = ["epsilon_greedy", "ucb1", "thompson"];
const ALGO_LABELS = { epsilon_greedy: "Epsilon-Greedy", ucb1: "UCB1", thompson: "Thompson Sampling" };

function useLiveBanditRace(running, resetToken) {
  const agentsRef = useRef({});
  const [regretHistory, setRegretHistory] = useState({ epsilon_greedy: [], ucb1: [], thompson: [] });
  const [estimates, setEstimates] = useState({ epsilon_greedy: [], ucb1: [], thompson: [] });
  const cumRegretRef = useRef({ epsilon_greedy: 0, ucb1: 0, thompson: 0 });
  const bestRev = Math.max(...PRICES.map(expectedRevenue));

  useEffect(() => {
    agentsRef.current = {};
    ALGO_KEYS.forEach((key) => {
      agentsRef.current[key] = new BANDIT_ALGORITHMS[key](PRICES.length);
    });
    cumRegretRef.current = { epsilon_greedy: 0, ucb1: 0, thompson: 0 };
    setRegretHistory({ epsilon_greedy: [], ucb1: [], thompson: [] });
    setEstimates({ epsilon_greedy: [], ucb1: [], thompson: [] });
  }, [resetToken]);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      setRegretHistory((prev) => {
        const next = { ...prev };
        ALGO_KEYS.forEach((key) => {
          const agent = agentsRef.current[key];
          const arm = agent.selectArm();
          const reward = PricingBanditEnvLike.pull(arm);
          agent.update(arm, reward);
          cumRegretRef.current[key] += bestRev - expectedRevenue(PRICES[arm]);
          next[key] = [...prev[key], cumRegretRef.current[key]];
        });
        return next;
      });
      setEstimates(() => {
        const next = {};
        ALGO_KEYS.forEach((key) => { next[key] = agentsRef.current[key].estimates(); });
        return next;
      });
    }, 60);
    return () => clearInterval(id);
  }, [running, bestRev]);

  return { regretHistory, estimates };
}

export default function BanditStage() {
  const { data } = useArtifact("bandit.json");
  const [running, setRunning] = useState(false);
  const [resetToken, setResetToken] = useState(0);
  const { regretHistory, estimates } = useLiveBanditRace(running, resetToken);
  const [selectedAlgo, setSelectedAlgo] = useState("thompson");

  return (
    <div>
      <h2 className="section-heading md-headline-small">Which price makes the most money?</h2>
      <p className="md-body-medium" style={{ color: "var(--md-sys-color-on-surface-variant)", marginBottom: "1rem", maxWidth: "70ch" }}>
        No inventory limit, no deadline yet - just a price to set, over and over, against an unknown demand curve.
        Three classic bandit algorithms explore that trade-off differently: how much do you keep testing other
        prices versus sticking with what's worked so far?
      </p>

      <div className="m3-card m3-card-filled" style={{ padding: "1.2rem", marginBottom: "1.5rem" }}>
        <div style={{ display: "flex", gap: "0.6rem", marginBottom: "1rem" }}>
          <button className="m3-button m3-button-filled md-label-large" onClick={() => setRunning((r) => !r)}>
            {running ? "Pause" : "Run live"}
          </button>
          <button
            className="m3-button m3-button-outlined md-label-large"
            onClick={() => { setRunning(false); setResetToken((t) => t + 1); }}
          >
            Reset
          </button>
        </div>
        <p className="md-label-medium" style={{ color: "var(--md-sys-color-on-surface-variant)", marginBottom: "0.5rem" }}>
          Live cumulative regret (this run, right now, in your browser)
        </p>
        <MiniLineChart
          series={ALGO_KEYS.map((key, i) => ({ name: ALGO_LABELS[key], color: SERIES_COLORS[i], data: regretHistory[key] }))}
          xLabel="round"
          yLabel="cumulative regret ($)"
          height={260}
        />

        <div style={{ marginTop: "1.2rem" }}>
          <div className="m3-segmented" style={{ marginBottom: "0.8rem" }}>
            {ALGO_KEYS.map((key) => (
              <button
                key={key}
                className={`m3-segment ${selectedAlgo === key ? "is-selected" : ""}`}
                onClick={() => setSelectedAlgo(key)}
              >
                {ALGO_LABELS[key]}
              </button>
            ))}
          </div>
          <p className="md-label-medium" style={{ color: "var(--md-sys-color-on-surface-variant)", marginBottom: "0.5rem" }}>
            {ALGO_LABELS[selectedAlgo]}'s current revenue estimate per price
          </p>
          <EstimateBars estimates={estimates[selectedAlgo] || []} />
        </div>
      </div>

      {data && (
        <div className="m3-card m3-card-elevated" style={{ padding: "1.2rem" }}>
          <h3 className="md-title-medium" style={{ marginBottom: "0.4rem" }}>Averaged over 40 runs of {data.n_rounds} rounds</h3>
          <p className="md-body-small" style={{ color: "var(--md-sys-color-on-surface-variant)", marginBottom: "0.8rem" }}>
            A single live run above is noisy - this is the same three algorithms, trained offline in Python, averaged
            over 40 independent runs. Epsilon-greedy eventually finds the right price too, but pays a much bigger
            "learning tax" getting there than UCB1 or Thompson Sampling do - the actual textbook reason those two
            exist.
          </p>
          <MiniLineChart
            series={ALGO_KEYS.map((key, i) => ({
              name: data.algorithms[key].name,
              color: SERIES_COLORS[i],
              data: data.algorithms[key].regret_curve,
            }))}
            xLabel="round (binned)"
            yLabel="cumulative regret ($)"
            height={260}
          />
        </div>
      )}
    </div>
  );
}

function EstimateBars({ estimates }) {
  if (!estimates.length) return <p className="empty-state md-body-medium">Run live to see estimates form.</p>;
  const max = Math.max(...estimates, 1);
  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: "6px", height: 110 }}>
      {PRICES.map((price, i) => (
        <div key={price} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
          <div
            style={{
              width: "100%",
              height: Math.max(2, (estimates[i] / max) * 90),
              background: "var(--md-sys-color-primary)",
              borderRadius: "4px 4px 0 0",
              transition: "height 0.15s ease",
            }}
          />
          <span className="md-label-small mono" style={{ color: "var(--md-sys-color-on-surface-variant)" }}>${price}</span>
        </div>
      ))}
    </div>
  );
}
