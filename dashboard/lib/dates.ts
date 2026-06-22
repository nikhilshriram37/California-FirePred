// Date helpers — all California (Pacific) oriented, string-based to avoid TZ drift.

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** Pacific calendar day as YYYY-MM-DD. */
export function pacificToday(): string {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/Los_Angeles", year: "numeric", month: "2-digit", day: "2-digit",
  }).format(new Date());
}

/** "Jun 21" from "2026-06-21" (no Date parsing, so no timezone drift). */
export function formatMD(iso?: string): string {
  if (!iso) return "—";
  const [, m, d] = iso.split("-");
  return `${MONTHS[Number(m) - 1]} ${Number(d)}`;
}

/** Whole-day difference (iso − todayIso), computed in UTC to avoid drift. */
export function dayDiff(iso: string, todayIso: string): number {
  const a = Date.parse(`${iso}T00:00:00Z`);
  const b = Date.parse(`${todayIso}T00:00:00Z`);
  return Math.round((a - b) / 86_400_000);
}

export function relLabel(diff: number): string {
  if (diff === 0) return "Today";
  if (diff === 1) return "Tomorrow";
  if (diff > 1) return `+${diff} days`;
  if (diff === -1) return "Yesterday";
  return `${-diff} days ago`;
}

/** "5m ago" / "3h ago" / "2d ago" from a timestamp. */
export function timeAgo(iso?: string | null): string {
  if (!iso) return "—";
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}
