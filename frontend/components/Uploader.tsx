"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";

export default function Uploader() {
  const router = useRouter();
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function send(file?: File) {
    if (!file || busy) return;
    setBusy(true);
    setError(null);
    try {
      const a = await api.upload(file);
      router.push(`/analyses/${a.id}`);
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  }

  return (
    <div>
      <div
        className={`intake ${drag ? "drag" : ""} ${busy ? "busy" : ""}`}
        role="button"
        tabIndex={0}
        onClick={() => input.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); send(e.dataTransfer.files[0]); }}
      >
        {busy ? (
          <>
            <div className="spinner ink" />
            <span className="intake-title">Uploading…</span>
          </>
        ) : (
          <>
            <svg className="intake-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"
                 strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <path d="M12 16V4M7 9l5-5 5 5M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
            </svg>
            <span className="intake-title">{drag ? "Drop to upload" : "Drop or click to upload"}</span>
            <span className="label">JPG, PNG, WebP · 25 MB</span>
          </>
        )}
        <input ref={input} type="file" accept="image/*" hidden onChange={(e) => send(e.target.files?.[0])} />
      </div>
      {error && <p className="error">{error}</p>}
    </div>
  );
}
