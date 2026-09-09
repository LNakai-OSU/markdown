import { useEffect, useState } from "react";

export function useArtifact(filename) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`/data/${filename}`)
      .then((res) => {
        if (!res.ok) throw new Error(`${filename}: ${res.status}`);
        return res.json();
      })
      .then(setData)
      .catch((e) => setError(e.message));
  }, [filename]);

  return { data, error };
}
