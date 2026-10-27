"use client";

import { useEffect, useState } from "react";
import { api, type Health } from "@/lib/api";

export default function ApiStatus() {
  const [health, setHealth] = useState<Health | null>(null);
  const [down, setDown] = useState(false);

  useEffect(() => {
    const check = () =>
      api.health().then((h) => { setHealth(h); setDown(false); }).catch(() => setDown(true));
    check();
    const t = setInterval(check, 30_000);
    return () => clearInterval(t);
  }, []);

  if (down) return <span className="status-dot down">API offline</span>;
  if (!health) return <span className="status-dot">Connecting</span>;

  const ready = health.sam_loaded;
  const title = `SAM ${health.sam_loaded ? "loaded" : "not loaded"} · classifier ${
    health.classifier_loaded ? "loaded" : "not loaded"
  }`;
  return (
    <span className={`status-dot ${ready ? "ok" : "down"}`} title={title}>
      {health.model_version ? `Model ${health.model_version}` : ready ? "Ready" : "Models missing"}
    </span>
  );
}
