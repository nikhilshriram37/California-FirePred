-- FireProject — initial schema for the live wildfire risk dashboard.
-- Apply with the Supabase CLI (`supabase db push`) or paste into the SQL editor.
--
-- Design:
--   grid_cells       static ~10km cells (the model's grid) + geometry
--   risk_scores      one row per cell per scored day (what the map draws)
--   feature_history  the streamed daily features (the dataset that grows)
--   risk_meta        per-run snapshot metadata (latest = what the dashboard shows)
--   active_fires     optional archive of FIRMS detections (dashboard reads FIRMS live)

create extension if not exists postgis;

-- ---------------------------------------------------------------------------
-- Static grid
-- ---------------------------------------------------------------------------
create table if not exists grid_cells (
  grid_id     integer primary key,
  lat_center  double precision not null,
  lon_center  double precision not null,
  geom        geometry(Point, 4326)
);
create index if not exists grid_cells_geom_idx on grid_cells using gist (geom);

-- ---------------------------------------------------------------------------
-- Daily risk scores (the choropleth)
-- ---------------------------------------------------------------------------
create table if not exists risk_scores (
  id               bigint generated always as identity primary key,
  grid_id          integer not null references grid_cells (grid_id),
  date             date not null,
  raw_probability  double precision,
  risk             double precision not null,
  tier             text not null check (tier in ('Red', 'Yellow', 'Green')),
  model_version    text,
  unique (grid_id, date)
);
create index if not exists risk_scores_date_idx on risk_scores (date);

-- ---------------------------------------------------------------------------
-- Streamed feature history — grows the training dataset over time.
-- Features stored as jsonb so the schema survives feature-set changes; has_fire
-- is backfilled later once a cell's outcome is known.
-- ---------------------------------------------------------------------------
create table if not exists feature_history (
  id        bigint generated always as identity primary key,
  grid_id   integer not null references grid_cells (grid_id),
  date      date not null,
  features  jsonb not null,
  has_fire  integer,
  unique (grid_id, date)
);
create index if not exists feature_history_date_idx on feature_history (date);

-- ---------------------------------------------------------------------------
-- Snapshot metadata (latest row = current dashboard state)
-- ---------------------------------------------------------------------------
create table if not exists risk_meta (
  id              bigint generated always as identity primary key,
  data_date       date not null,
  generated_at    timestamptz not null default now(),
  source          text,
  mode            text,
  model_version   text,
  n_cells         integer,
  tier_counts     jsonb,
  thresholds      jsonb,
  actual_fires    integer,
  lightning_cells integer
);
create index if not exists risk_meta_data_date_idx on risk_meta (data_date desc);

-- ---------------------------------------------------------------------------
-- Optional FIRMS archive (dashboard fetches FIRMS live; this is for history)
-- ---------------------------------------------------------------------------
create table if not exists active_fires (
  id          bigint generated always as identity primary key,
  latitude    double precision not null,
  longitude   double precision not null,
  frp         double precision,
  confidence  text,
  acq_date    date,
  acq_time    text,
  satellite   text,
  geom        geometry(Point, 4326),
  ingested_at timestamptz not null default now()
);
create index if not exists active_fires_acq_date_idx on active_fires (acq_date);

-- ---------------------------------------------------------------------------
-- Row-level security: public read on display tables; writes via service role
-- (which bypasses RLS), so no write policies are defined.
-- ---------------------------------------------------------------------------
alter table grid_cells      enable row level security;
alter table risk_scores     enable row level security;
alter table feature_history enable row level security;
alter table risk_meta       enable row level security;
alter table active_fires    enable row level security;

do $$
begin
  create policy "public read grid_cells"  on grid_cells      for select using (true);
  create policy "public read risk_scores" on risk_scores     for select using (true);
  create policy "public read risk_meta"   on risk_meta       for select using (true);
  create policy "public read fires"       on active_fires    for select using (true);
exception when duplicate_object then null;
end $$;
