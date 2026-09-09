import { useEffect, useRef, useState } from "react";
import { RevenueManagementEnv, N_DAYS, START_INVENTORY } from "../lib/revenueEnv";
import MiniLineChart from "./MiniLineChart";

// Plays two policies through independent, freshly-randomized episodes of
// the same environment, side by side, one simulated day at a time - so you
// can watch two pricing strategies actually diverge rather than just
// comparing two final numbers.
export default function EpisodePlayer({ policies, richState = false, stepMs = 180 }) {
  const [runId, setRunId] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [frames, setFrames] = useState(policies.map(() => []));
  const [finished, setFinished] = useState(policies.map(() => false));
  const envsRef = useRef([]);

  useEffect(() => {
    envsRef.current = policies.map(() => new RevenueManagementEnv(richState));
    reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, richState]);

  function reset() {
    envsRef.current.forEach((env) => env.reset());
    setFrames(policies.map(() => [{ day: 0, inventory: START_INVENTORY, revenue: 0, price: null }]));
    setFinished(policies.map(() => false));
  }

  useEffect(() => {
    if (!playing) return;
    let ticking = false;
    const id = setInterval(async () => {
      if (ticking) return; // guard against a slow (async) policyFn overlapping the next tick
      ticking = true;
      try {
        const actions = await Promise.all(
          policies.map((p, idx) => {
            const env = envsRef.current[idx];
            return env.done ? null : Promise.resolve(p.policyFn(env.state()));
          })
        );
        setFrames((prev) =>
          prev.map((series, idx) => {
            const env = envsRef.current[idx];
            if (env.done || actions[idx] == null) return series;
            const { reward, price } = env.step(actions[idx]);
            const last = series[series.length - 1];
            return [...series, {
              day: N_DAYS - env.daysRemaining,
              inventory: env.inventory,
              revenue: last.revenue + reward,
              price,
            }];
          })
        );
        setFinished(envsRef.current.map((env) => env.done));
      } finally {
        ticking = false;
      }
    }, stepMs);
    return () => clearInterval(id);
  }, [playing, stepMs, policies]);

  useEffect(() => {
    if (finished.length && finished.every(Boolean)) setPlaying(false);
  }, [finished]);

  const revenueSeries = policies.map((p, i) => ({
    name: p.name,
    color: p.color,
    data: frames[i].map((f) => f.revenue),
  }));
  const inventorySeries = policies.map((p, i) => ({
    name: p.name,
    color: p.color,
    data: frames[i].map((f) => f.inventory),
  }));

  return (
    <div>
      <div style={{ display: "flex", gap: "0.6rem", marginBottom: "1rem", alignItems: "center" }}>
        <button className="m3-button m3-button-filled md-label-large" onClick={() => setPlaying((p) => !p)}>
          {playing ? "Pause" : "Play episode"}
        </button>
        <button className="m3-button m3-button-outlined md-label-large" onClick={() => setRunId((n) => n + 1)}>
          New random episode
        </button>
        {policies.map((p, i) => {
          const f = frames[i][frames[i].length - 1];
          return (
            <span key={p.name} className="m3-chip md-label-large" style={{ borderColor: p.color }}>
              <span style={{ width: 8, height: 8, borderRadius: "50%", background: p.color, display: "inline-block" }} />
              {p.name}: {f?.price ? `$${f.price}` : "-"} · {f?.inventory ?? START_INVENTORY} left
            </span>
          );
        })}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.2rem" }}>
        <div>
          <p className="md-label-medium" style={{ color: "var(--md-sys-color-on-surface-variant)", marginBottom: 4 }}>Cumulative revenue</p>
          <MiniLineChart series={revenueSeries} xLabel="day" yLabel="$" height={220} />
        </div>
        <div>
          <p className="md-label-medium" style={{ color: "var(--md-sys-color-on-surface-variant)", marginBottom: 4 }}>Inventory remaining</p>
          <MiniLineChart series={inventorySeries} xLabel="day" yLabel="units" height={220} />
        </div>
      </div>
    </div>
  );
}
