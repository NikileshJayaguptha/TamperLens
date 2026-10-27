import { scoreColor, verdict } from "@/lib/api";

const STATUS_COLOR: Record<string, string> = {
  pending: "var(--ink-2)",
  processing: "var(--amber)",
  failed: "var(--ai)",
};

const STATUS_TEXT: Record<string, string> = {
  pending: "Queued",
  processing: "Processing",
  failed: "Failed",
};

export default function ScoreBadge({ score, status }: { score: number | null; status?: string }) {
  if (status && status !== "done") {
    return (
      <span className="tag" style={{ color: STATUS_COLOR[status] ?? "var(--ink-2)" }}>
        {STATUS_TEXT[status] ?? status}
      </span>
    );
  }
  if (score == null) return <span className="tag" style={{ color: "var(--ink-3)" }}>No regions</span>;
  return <span className="tag" style={{ color: scoreColor(score) }}>{verdict(score)}</span>;
}
