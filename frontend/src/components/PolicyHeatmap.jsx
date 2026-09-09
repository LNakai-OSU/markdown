import { useEffect, useRef, useState } from "react";
import { PRICE_MIN, PRICE_MAX } from "../lib/demand";

// A single-hue sequential ramp (light -> dark = cheap -> expensive), per
// the "sequential data gets one hue" rule - never a rainbow for magnitude.
// Interpolates in a rough OKLab-ish way (lerp in sRGB is good enough at
// this size) between the primary color's light and dark tonal steps.
const RAMP_LOW = [222, 226, 255]; // matches --md-sys-color-primary-container-ish light step
const RAMP_HIGH = [35, 42, 96]; // deep indigo

function priceColor(price) {
  const t = (price - PRICE_MIN) / (PRICE_MAX - PRICE_MIN);
  const rgb = RAMP_LOW.map((lo, i) => Math.round(lo + (RAMP_HIGH[i] - lo) * t));
  return `rgb(${rgb.join(",")})`;
}

export default function PolicyHeatmap({ grid, title, maxInventory = 40 }) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [hover, setHover] = useState(null);

  const nDays = grid.length; // rows: day index 0..N_DAYS, we render 1..N_DAYS (0 is terminal, no decision)
  const nInv = grid[0]?.length || 0;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const rows = nDays - 1; // skip day 0 (terminal, no pricing decision)
    const cols = nInv - 1; // skip inventory 0 (nothing left to sell)
    canvas.width = cols;
    canvas.height = rows;

    for (let d = 1; d < nDays; d++) {
      for (let i = 1; i < nInv; i++) {
        const price = grid[d][i];
        ctx.fillStyle = priceColor(price);
        // row 0 at the top should be the most days remaining
        const rowY = rows - d;
        ctx.fillRect(i - 1, rowY, 1, 1);
      }
    }
  }, [grid, nDays, nInv]);

  function handleMove(e) {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor(((e.clientX - rect.left) / rect.width) * (nInv - 1)) + 1;
    const y = Math.floor(((e.clientY - rect.top) / rect.height) * (nDays - 1));
    const day = nDays - 1 - y;
    if (day < 1 || day > nDays - 1 || x < 1 || x > nInv - 1) {
      setHover(null);
      return;
    }
    setHover({ day, inventory: x, price: grid[day][x] });
  }

  return (
    <div>
      {title && <p className="md-label-large" style={{ marginBottom: "0.5rem", color: "var(--md-sys-color-on-surface-variant)" }}>{title}</p>}
      <div ref={containerRef} style={{ display: "flex", gap: "0.6rem" }}>
        <div style={{ display: "flex", flexDirection: "column", justifyContent: "space-between", fontSize: 11, color: "var(--md-sys-color-on-surface-variant)", height: 220 }}>
          <span>{nDays - 1}d left</span>
          <span>1d left</span>
        </div>
        <div style={{ position: "relative", flex: 1 }}>
          <canvas
            ref={canvasRef}
            onMouseMove={handleMove}
            onMouseLeave={() => setHover(null)}
            style={{
              width: "100%",
              height: 220,
              imageRendering: "pixelated",
              borderRadius: "var(--md-sys-shape-corner-small)",
              display: "block",
              cursor: "crosshair",
            }}
          />
          {hover && (
            <div
              className="mono"
              style={{
                position: "absolute",
                top: 6,
                right: 6,
                background: "var(--md-sys-color-inverse-surface)",
                color: "var(--md-sys-color-inverse-on-surface)",
                padding: "4px 8px",
                borderRadius: 6,
                fontSize: 11,
                pointerEvents: "none",
              }}
            >
              day {hover.day}, {hover.inventory} left &rarr; ${hover.price}
            </div>
          )}
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--md-sys-color-on-surface-variant)", marginTop: 4 }}>
            <span>0 units left</span>
            <span>{maxInventory} units left</span>
          </div>
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginTop: "0.6rem" }}>
        <span className="md-label-small" style={{ color: "var(--md-sys-color-on-surface-variant)" }}>${PRICE_MIN}</span>
        <div style={{ flex: 1, height: 8, borderRadius: 4, background: `linear-gradient(90deg, ${priceColor(PRICE_MIN)}, ${priceColor(PRICE_MAX)})` }} />
        <span className="md-label-small" style={{ color: "var(--md-sys-color-on-surface-variant)" }}>${PRICE_MAX}</span>
      </div>
    </div>
  );
}
