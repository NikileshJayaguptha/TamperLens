"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import Uploader from "@/components/Uploader";
import ScoreBadge from "@/components/ScoreBadge";
import { api, pct, scoreColor, type AnalysisSummary, type LabelStats } from "@/lib/api";

export default function Home() {
  const [items, setItems] = useState<AnalysisSummary[] | null>(null);
  const [stats, setStats] = useState<LabelStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.list(), api.stats()])
      .then(([list, s]) => { setItems(list); setStats(s); })
      .catch((e) => setError(`Can't reach the API: ${e.message}`));
  }, []);

  const working = items?.some((a) => a.status === "pending" || a.status === "processing");
  useEffect(() => {
    if (!working) return;
    const t = setInterval(() => api.list().then(setItems).catch(() => {}), 2000);
    return () => clearInterval(t);
  }, [working]);

  const labelled = stats ? stats.ai + stats.real : 0;
  const totalRegions = stats ? labelled + stats.unlabeled : 0;

  return (
    <>
      <section className="intro">
        <h1>Detect AI edits</h1>
      </section>

      <Uploader />

      {error && <p className="error">{error}</p>}

      {stats && (
        <div className="stats">
          <span><b>{items?.length ?? 0}</b> analyses</span>
          <span><b>{totalRegions}</b> regions</span>
          <span><b>{labelled}</b> labelled</span>
        </div>
      )}

      <div className="section-head">
        <h2>Recent</h2>
      </div>

      {items && items.length === 0 && <div className="empty">No analyses yet.</div>}
      {!items && !error && <div className="center"><div className="spinner ink" /></div>}

      <div className="cases">
        {items?.map((a) => (
          <Link key={a.id} href={`/analyses/${a.id}`} className="case">
            <div className="thumb">
              <img src={api.imageUrl(a.id)} alt={a.filename} loading="lazy" />
              {a.status === "done" && <img className="heat" src={api.heatmapUrl(a.id)} alt="" loading="lazy" />}
            </div>
            <div className="case-body">
              <div style={{ minWidth: 0 }}>
                <div className="case-name" title={a.filename}>{a.filename}</div>
                <div className="case-meta">
                  {new Date(a.created_at).toLocaleDateString(undefined, { day: "numeric", month: "short" })}
                </div>
              </div>
              {a.status === "done" && a.overall_score != null ? (
                <span className="case-score" style={{ color: scoreColor(a.overall_score) }}>{pct(a.overall_score)}</span>
              ) : (
                <ScoreBadge score={a.overall_score} status={a.status} />
              )}
            </div>
          </Link>
        ))}
      </div>
    </>
  );
}
