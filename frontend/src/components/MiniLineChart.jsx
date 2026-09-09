import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

// Fixed categorical order reused across every chart in the app, so a given
// series color always means the same thing (e.g. the bandit stage's
// "Thompson Sampling" line is the same hue everywhere it appears).
export const SERIES_COLORS = [
  "var(--md-sys-color-primary)",
  "var(--md-sys-color-tertiary)",
  "var(--md-sys-color-good)",
  "var(--md-sys-color-error)",
];

export default function MiniLineChart({ series, xLabel, yLabel, height = 260, valueFormatter }) {
  const maxLen = Math.max(...series.map((s) => s.data.length));
  const merged = [];
  for (let i = 0; i < maxLen; i++) {
    const point = { x: i };
    series.forEach((s) => {
      if (s.data[i] !== undefined) point[s.name] = s.data[i];
    });
    merged.push(point);
  }

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <LineChart data={merged} margin={{ top: 8, right: 12, bottom: 8, left: 4 }}>
          <CartesianGrid stroke="var(--md-sys-color-outline-variant)" strokeOpacity={0.35} vertical={false} />
          <XAxis
            dataKey="x"
            stroke="var(--md-sys-color-outline)"
            tick={{ fontSize: 11, fill: "var(--md-sys-color-on-surface-variant)" }}
            label={{ value: xLabel, position: "insideBottom", offset: -4, fontSize: 11, fill: "var(--md-sys-color-outline)" }}
          />
          <YAxis
            stroke="var(--md-sys-color-outline)"
            tick={{ fontSize: 11, fill: "var(--md-sys-color-on-surface-variant)" }}
            width={54}
            label={{ value: yLabel, angle: -90, position: "insideLeft", fontSize: 11, fill: "var(--md-sys-color-outline)" }}
          />
          <Tooltip
            contentStyle={{
              background: "var(--md-sys-color-inverse-surface)",
              border: "none",
              borderRadius: 8,
              fontSize: 12,
            }}
            labelStyle={{ color: "var(--md-sys-color-inverse-on-surface)" }}
            itemStyle={{ color: "var(--md-sys-color-inverse-on-surface)" }}
            formatter={valueFormatter}
          />
          {series.length > 1 && <Legend wrapperStyle={{ fontSize: 12 }} />}
          {series.map((s, i) => (
            <Line
              key={s.name}
              type="monotone"
              dataKey={s.name}
              stroke={s.color || SERIES_COLORS[i % SERIES_COLORS.length]}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
