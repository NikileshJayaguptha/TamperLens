export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Label = "real" | "ai" | null;

export interface Region {
  id: number;
  index: number;
  bbox: [number, number, number, number];
  polygons: number[][][];
  area_frac: number;
  noise_score: number;
  classifier_score: number | null;
  score: number;
  details: { noise_z?: number; ela_z?: number };
  label: Label;
}

export interface AnalysisSummary {
  id: string;
  filename: string;
  status: "pending" | "processing" | "done" | "failed";
  error: string | null;
  width: number;
  height: number;
  overall_score: number | null;
  model_version: string | null;
  created_at: string;
}

export interface Analysis extends AnalysisSummary {
  regions: Region[];
}

export interface Health {
  status: string;
  sam_loaded: boolean;
  classifier_loaded: boolean;
  model_version: string | null;
}

export interface LabelStats {
  real: number;
  ai: number;
  unlabeled: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { cache: "no-store", ...init });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {}
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

export const api = {
  list: () => request<AnalysisSummary[]>("/api/analyses"),
  get: (id: string) => request<Analysis>(`/api/analyses/${id}`),
  remove: (id: string) => request<void>(`/api/analyses/${id}`, { method: "DELETE" }),
  upload: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<AnalysisSummary>("/api/analyses", { method: "POST", body });
  },
  label: (regionId: number, label: Label) =>
    request<Region>(`/api/regions/${regionId}/label`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label }),
    }),
  stats: () => request<LabelStats>("/api/labels/stats"),
  health: () => request<Health>("/api/health"),
  imageUrl: (id: string) => `${API_URL}/api/analyses/${id}/image`,
  heatmapUrl: (id: string) => `${API_URL}/api/analyses/${id}/heatmap`,
};

const RAMP: [number, number, number][] = [[43, 122, 86], [201, 139, 18], [224, 70, 27]];

export function scoreColor(score: number): string {
  const t = Math.min(Math.max(score, 0), 1) * (RAMP.length - 1);
  const i = Math.min(Math.floor(t), RAMP.length - 2);
  const f = t - i;
  const [r, g, b] = RAMP[i].map((c, k) => Math.round(c + (RAMP[i + 1][k] - c) * f));
  return `rgb(${r} ${g} ${b})`;
}

export function verdict(score: number): string {
  if (score >= 0.7) return "Likely AI-edited";
  if (score >= 0.4) return "Inconclusive";
  return "Likely authentic";
}

export function caseNo(id: string): string {
  return id.replace(/-/g, "").slice(0, 6).toUpperCase();
}

export function pct(v: number | null | undefined): string {
  return v == null ? "–" : `${Math.round(v * 100)}%`;
}

export function areaPct(v: number): string {
  return v < 0.01 ? `${(v * 100).toFixed(1)}%` : pct(v);
}
