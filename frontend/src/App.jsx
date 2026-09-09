import { useState } from "react";
import BanditStage from "./components/BanditStage";
import QLearningStage from "./components/QLearningStage";
import DQNStage from "./components/DQNStage";
import { N_DAYS, START_INVENTORY } from "./lib/revenueEnv";

const TABS = [
  { id: "bandit", label: "Stage 1 · Bandits" },
  { id: "qlearning", label: "Stage 2 · Q-Learning" },
  { id: "dqn", label: "Stage 3 · Deep Q-Network" },
];

export default function App() {
  const [tab, setTab] = useState("bandit");

  return (
    <div className="app-shell">
      <div className="app-header">
        <div className="logo-mark">
          <svg width="24" height="24" viewBox="0 0 32 32">
            <path
              d="M7 21 L13 13 L18 17 L25 8"
              stroke="var(--md-sys-color-on-primary-container)"
              strokeWidth="2.6"
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
            />
          </svg>
        </div>
        <div>
          <h1 className="md-headline-medium">Markdown</h1>
          <p className="tagline md-body-medium">
            Three agents, one pricing problem: sell {START_INVENTORY} units in {N_DAYS} days. Watch each one learn.
          </p>
        </div>
      </div>

      <div className="disclaimer md-body-medium">
        <span>
          A teaching project on reinforcement learning, not a real pricing engine: the market (demand curve,
          competitor behavior) is simulated, and every agent here is trained offline - what you're watching is a
          saved policy playing out a fresh, randomly-drawn episode live in your browser, not live training.
        </span>
      </div>

      <div className="tab-nav">
        <div className="m3-tabs">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={`m3-tab ${tab === t.id ? "is-active" : ""}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {tab === "bandit" && <BanditStage />}
      {tab === "qlearning" && <QLearningStage />}
      {tab === "dqn" && <DQNStage />}
    </div>
  );
}
