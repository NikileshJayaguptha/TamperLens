"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import ScoreBadge from "@/components/ScoreBadge";
import { api, areaPct, pct, scoreColor, verdict, type Analysis, type Label, type Region } from "@/lib/api";

type Filter = "all" | "flagged" | "unlabelled" | "labelled";

const FILTERS: [Filter, string][] = [
  ["all", "All"],
  ["flagged", "Flagged"],
  ["unlabelled", "Unlabelled"],
  ["labelled", "Labelled"],
];

export default function AnalysisPage({ params }: { params: { id: string } }) {
  const router = useRouter();
  const [data, setData] = useState<Analysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);
  const [showHeat, setShowHeat] = useState(true);
  const [showOutlines, setShowOutlines] = useState(true);
  const [heatOpacity, setHeatOpacity] = useState(0.6);
  const [threshold, setThreshold] = useState(0.3);
  const [filter, setFilter] = useState<Filter>("all");

  const load = useCallback(() => {
    api.get(params.id).then(setData).catch((e) => setError(e.message));
  }, [params.id]);

  useEffect(load, [load]);

  useEffect(() => {
    if (!data || data.status === "done" || data.status === "failed") return;
    const t = setInterval(load, 1500);
    return () => clearInterval(t);
  }, [data, load]);

  const regions = data?.regions ?? [];

  const visible = useMemo(
    () => regions.filter((r) => r.score >= threshold || r.id === selected || r.id === hovered || r.label),
    [regions, threshold, selected, hovered],
  );

  const listed = useMemo(() => {
    const keep = (r: Region) =>
      filter === "flagged" ? r.score >= threshold
      : filter === "unlabelled" ? !r.label
      : filter === "labelled" ? !!r.label
      : true;
    return regions.filter(keep).sort((a, b) => b.score - a.score);
  }, [regions, filter, threshold]);

  const labelledCount = regions.filter((r) => r.label).length;
  const flaggedCount = regions.filter((r) => r.score >= threshold).length;

  const setLabel = useCallback(async (region: Region, label: Label) => {
    const next = region.label === label ? null : label;
    try {
      const updated = await api.label(region.id, next);
      setData((d) => d && { ...d, regions: d.regions.map((r) => (r.id === updated.id ? updated : r)) });
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.metaKey || e.ctrlKey || e.altKey) return;
      if (!listed.length) return;
      const idx = listed.findIndex((r) => r.id === selected);
      const current = idx >= 0 ? listed[idx] : null;
      const move = (to: number) => {
        e.preventDefault();
        setSelected(listed[Math.max(0, Math.min(listed.length - 1, to))].id);
      };
      if (e.key === "j" || e.key === "ArrowDown") move(idx + 1);
      else if (e.key === "k" || e.key === "ArrowUp") move(idx < 0 ? 0 : idx - 1);
      else if (e.key === "Escape") setSelected(null);
      else if (current && e.key === "r") setLabel(current, "real");
      else if (current && e.key === "a") setLabel(current, "ai");
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [listed, selected, setLabel]);

  useEffect(() => {
    if (selected != null) document.getElementById(`region-${selected}`)?.scrollIntoView({ block: "nearest" });
  }, [selected]);

  async function remove() {
    if (!confirm("Delete this case and its labels?")) return;
    await api.remove(params.id);
    router.push("/");
  }

  if (error && !data) return <p className="error">{error}</p>;
  if (!data) return <div className="center"><div className="spinner ink" /></div>;

  const working = data.status === "pending" || data.status === "processing";
  const done = data.status === "done";
  const tagSize = Math.max(14, data.width * 0.018);

  return (
    <>
      <Link href="/" className="back">← Back</Link>

      <header className="case-header">
        <div>
          <h1>{data.filename}</h1>
          <div className="facts">
            <span>{data.width}×{data.height}px</span>
            <span>{new Date(data.created_at).toLocaleDateString(undefined, { dateStyle: "medium" })}</span>
            {!done && <ScoreBadge score={data.overall_score} status={data.status} />}
          </div>
        </div>

        {done && data.overall_score != null && (
          <div className="verdict">
            <span className="label">Top score</span>
            <div className="verdict-row" style={{ marginTop: 4 }}>
              <span className="pct" style={{ color: scoreColor(data.overall_score) }}>{pct(data.overall_score)}</span>
              <span className="word">{verdict(data.overall_score)}</span>
            </div>
            <div className="meter">
              <span style={{ width: pct(data.overall_score), background: scoreColor(data.overall_score) }} />
            </div>
          </div>
        )}
      </header>

      {error && <p className="error">{error}</p>}
      {data.status === "failed" && <p className="error">Analysis failed: {data.error}</p>}

      <div className="workspace">
        <section>
          <div className="panel">
            <div className="toolbar">
              <div className="seg" role="group" aria-label="Layers">
                <button className={showHeat ? "on" : ""} onClick={() => setShowHeat(!showHeat)} aria-pressed={showHeat}>Heatmap</button>
                <button className={showOutlines ? "on" : ""} onClick={() => setShowOutlines(!showOutlines)} aria-pressed={showOutlines}>Outlines</button>
              </div>
              <label className="slider">
                Opacity
                <input type="range" min={0} max={1} step={0.05} value={heatOpacity} disabled={!showHeat}
                       onChange={(e) => setHeatOpacity(+e.target.value)} />
                <output>{pct(heatOpacity)}</output>
              </label>
              <label className="slider">
                Threshold
                <input type="range" min={0} max={1} step={0.05} value={threshold}
                       onChange={(e) => setThreshold(+e.target.value)} />
                <output>{pct(threshold)}</output>
              </label>
            </div>

            <div className="viewer">
              <img src={api.imageUrl(data.id)} alt={data.filename} />
              {done && (
                <img className="overlay" src={api.heatmapUrl(data.id)} alt=""
                     style={{ opacity: showHeat ? heatOpacity : 0 }} />
              )}
              {done && showOutlines && (
                <svg viewBox={`0 0 ${data.width} ${data.height}`} preserveAspectRatio="none">
                  {visible.map((r) => {
                    const active = r.id === selected || r.id === hovered;
                    const stroke = r.label === "real" ? "var(--real)" : r.label === "ai" ? "var(--ai)" : scoreColor(r.score);
                    return (
                      <g key={r.id}
                         onClick={() => setSelected(r.id)}
                         onMouseEnter={() => setHovered(r.id)}
                         onMouseLeave={() => setHovered(null)}>
                        {r.polygons.map((poly, i) => (
                          <polygon
                            key={i}
                            points={poly.map(([x, y]) => `${x},${y}`).join(" ")}
                            fill={stroke}
                            fillOpacity={r.id === selected ? 0.25 : active ? 0.12 : 0}
                            stroke={stroke}
                            strokeWidth={active ? 3 : 1.5}
                            strokeDasharray={r.label ? undefined : "6 4"}
                            vectorEffect="non-scaling-stroke"
                          >
                            <title>Region {r.index + 1}: {pct(r.score)}</title>
                          </polygon>
                        ))}
                        {active && (
                          <g transform={`translate(${r.bbox[0]},${r.bbox[1]})`}>
                            <rect width={tagSize * 2.2} height={tagSize * 1.4} rx={tagSize * 0.3} fill={stroke} />
                            <text className="region-tag" x={tagSize * 1.1} y={tagSize} fontSize={tagSize}
                                  textAnchor="middle" fill="#fff">
                              {r.index + 1}
                            </text>
                          </g>
                        )}
                      </g>
                    );
                  })}
                </svg>
              )}
              {working && (
                <div className="viewer-busy">
                  <div className="spinner" />
                  <span>{data.status === "pending" ? "Queued" : "Analysing…"}</span>
                </div>
              )}
            </div>
          </div>

          <div className="legend">
            <span><i className="dash" style={{ borderColor: "var(--amber)" }} />Unreviewed</span>
            <span><i style={{ borderColor: "var(--ai)" }} />AI</span>
            <span><i style={{ borderColor: "var(--real)" }} />Real</span>
            <span className="keys">
              <kbd>j</kbd>/<kbd>k</kbd> · <kbd>r</kbd> · <kbd>a</kbd>
            </span>
          </div>
        </section>

        <aside className="panel findings">
          <div className="findings-head">
            <div className="findings-title">
              <h2>{flaggedCount}/{regions.length} flagged</h2>
              <button className="btn danger" onClick={remove}>Delete</button>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 8 }} className="label">
              <span>Reviewed</span><span>{labelledCount}/{regions.length}</span>
            </div>
            <div className="progress">
              <span style={{ width: regions.length ? `${(labelledCount / regions.length) * 100}%` : 0 }} />
            </div>
            <div className="filters">
              {FILTERS.map(([key, name]) => (
                <button key={key} className={`chip ${filter === key ? "on" : ""}`} onClick={() => setFilter(key)}>
                  {name}
                </button>
              ))}
            </div>
          </div>

          <div className="region-list">
            {listed.map((r) => (
              <div
                key={r.id}
                id={`region-${r.id}`}
                className={`region ${r.id === selected ? "selected" : ""}`}
                onClick={() => setSelected(r.id === selected ? null : r.id)}
                onMouseEnter={() => setHovered(r.id)}
                onMouseLeave={() => setHovered(null)}
              >
                <div className="region-head">
                  <span className="region-title">
                    Region {r.index + 1}<small>{areaPct(r.area_frac)}</small>
                  </span>
                  <span className="region-pct" style={{ color: scoreColor(r.score) }}>{pct(r.score)}</span>
                </div>
                <div className="bar">
                  <span style={{ width: pct(r.score), background: scoreColor(r.score) }} />
                  <i style={{ left: pct(threshold) }} />
                </div>
                <div className="metrics">
                  <span>Classifier <b>{pct(r.classifier_score)}</b></span>
                  <span>Noise <b>{pct(r.noise_score)}</b></span>
                </div>
                <div className="label-buttons">
                  <button className={`btn real ${r.label === "real" ? "on" : ""}`}
                          onClick={(e) => { e.stopPropagation(); setLabel(r, "real"); }}>
                    Real {r.id === selected && <kbd>r</kbd>}
                  </button>
                  <button className={`btn ai ${r.label === "ai" ? "on" : ""}`}
                          onClick={(e) => { e.stopPropagation(); setLabel(r, "ai"); }}>
                    AI {r.id === selected && <kbd>a</kbd>}
                  </button>
                </div>
              </div>
            ))}
            {done && regions.length === 0 && (
              <p className="muted small" style={{ padding: 16 }}>No regions found.</p>
            )}
            {done && regions.length > 0 && listed.length === 0 && (
              <p className="muted small" style={{ padding: 16 }}>No matches.</p>
            )}
            {working && <div className="center"><div className="spinner ink" /></div>}
          </div>
        </aside>
      </div>
    </>
  );
}
