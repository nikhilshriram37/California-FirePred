"use client";

const DAYS = [0, 1, 2, 3, 4, 5];

function label(h: number): string {
  if (h === 0) return "Today";
  return `+${h}d`;
}

export default function DaySelector({
  horizon, onSelect,
}: { horizon: number; onSelect: (h: number) => void }) {
  return (
    <div className="day-selector">
      <span className="day-selector-label">Forecast</span>
      {DAYS.map((h) => (
        <button
          key={h}
          className={`day-btn ${h === horizon ? "active" : ""}`}
          onClick={() => onSelect(h)}
          title={h === 0 ? "Today's forecast" : `${h}-day forecast`}
        >
          {label(h)}
        </button>
      ))}
    </div>
  );
}
