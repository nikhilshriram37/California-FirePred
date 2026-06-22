"use client";

import { formatMD } from "@/lib/dates";

// -1 = active fires (observations only); 0 = today; 1..5 = forecast days ahead.
const MODES = [-1, 0, 1, 2, 3, 4, 5];

export default function DaySelector({
  horizon, dates, today, onSelect,
}: {
  horizon: number;
  dates: Record<number, string>;
  today: string;
  onSelect: (h: number) => void;
}) {
  return (
    <div className="day-selector">
      {MODES.map((h) => {
        if (h === -1) {
          return (
            <button
              key={h}
              className={`day-btn fires ${h === horizon ? "active" : ""}`}
              onClick={() => onSelect(h)}
              title="Live satellite fire detections"
            >
              🔥 Fires
            </button>
          );
        }
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
    </div>
  );
}
