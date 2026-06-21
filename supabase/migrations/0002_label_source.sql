-- Track WHICH source confirmed each fire label, so retraining can prefer
-- high-fidelity confirmed-incident labels (IRWIN/CAL FIRE) over heat-only (FIRMS).
-- Values: 'irwin', 'calfire', 'firms', or '+'-joined combos (e.g. 'irwin+firms').
-- NULL on no-fire cells.
alter table feature_history add column if not exists label_source text;
