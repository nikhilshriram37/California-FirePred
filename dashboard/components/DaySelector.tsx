"use client";

import { formatMD } from "@/lib/dates";

// 0 = today; 1..5 = forecast days ahead. Active fires is an independent overlay
// toggle (see the 🔥 button), not a day.
const DAYS = [0, 1, 2, 3, 4, 5];

export default function DaySelector({
  horizon, dates, today, onSelect, showFires, onToggleFires, fireCount,
}: {
  horizon: number;
  dates: Record<number, string>;
  today: string;
  onSelect: (h: number) => void;
  showFires: boolean;
  onToggleFires: () => void;
  fireCount: number;
}) {
  return (
    <div className="day-selector">
      {DAYS.map((h) => {
        const date = dates[h];
        const isToday = date === today;
        return (
          <button
            key={h}
            className={`day-btn ${h === horizon ? "active" : ""} ${isToday ? "today" : ""}`}
            onClick={() => onSelect(h)}
            title={date ? `Forecast for ${date}` : `+${h}-day forecast`}
          >
            {date ? formatMD(date) : `+${h}d`}
            {isToday && <span className="today-tag">today</span>}
          </button>
        );
      })}
      <span className="day-sep" aria-hidden />
      <button
        className={`day-btn fires ${showFires ? "active" : ""}`}
        onClick={onToggleFires}
        title="Overlay live satellite fire detections on top of the risk map"
        aria-pressed={showFires}
      >
        🔥 Fires{fireCount ? ` · ${fireCount}` : ""}
      </button>
    </div>
  );
}
