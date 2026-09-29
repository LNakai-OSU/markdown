import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";
import { normalizeState } from "./revenueEnv";

export function useDqnModel() {
  const sessionRef = useRef(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    ort.InferenceSession.create(`${import.meta.env.BASE_URL}data/dqn_model.onnx`)
      .then((session) => {
        if (!cancelled) {
          sessionRef.current = session;
          setReady(true);
        }
      })
      .catch((e) => setError(e.message));
    return () => {
      cancelled = true;
    };
  }, []);

  async function predict(state) {
    const session = sessionRef.current;
    if (!session) return 0;
    const input = normalizeState(state);
    const tensor = new ort.Tensor("float32", input, [1, 4]);
    const results = await session.run({ state: tensor });
    const qValues = results.q_values.data;
    let best = 0;
    for (let i = 1; i < qValues.length; i++) {
      if (qValues[i] > qValues[best]) best = i;
    }
    return best;
  }

  return { ready, error, predict };
}
