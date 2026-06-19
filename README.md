# Oil Port Classifier

Classify ports/terminals as **LOAD / DISCHARGE / BOTH / STS** by combining
static reference data (World Port Index, Global Energy Monitor, OpenStreetMap)
with AIS-derived vessel behaviour (stationary periods, voyage sequencing,
ship-to-ship proximity).

## Feasibility summary

**Is it feasible?** Yes - via terminal-role seeding + voyage-sequence
alternation, NOT via draught (see below).

> **Important pivot from the original plan**: AISstream's
> `ShipStaticData.MaximumStaticDraught` is a **static hull constant**
> (the vessel's configured max draught) - it does not change between
> laden and ballast voyages, so it cannot tell us "did this vessel get
> heavier/lighter at this port?". Draught-delta classification has been
> removed from the main pipeline (kept as a LEGACY option in
> `classify.py` for anyone using a different AIS feed that *does* report
> dynamic draught).

The core question ("was this port call a LOAD or a DISCHARGE?") is still
tractable, because of one physical fact: **a tanker's cargo state
alternates with its port calls** - LOAD (becomes laden) -> sail laden ->
DISCHARGE (becomes ballast) -> sail ballast -> LOAD -> ... If we know the
role of even a handful of terminals in a vessel's itinerary (from GEM
data, or your own domain knowledge), the alternation rule fills in the
rest of that vessel's calls - and once enough vessels have resolved a
given terminal, that terminal's label can help classify other vessels'
otherwise-unresolved visits there too.

Not feasible as a "100% accurate label every port" tool without manual
spot-checking - but very workable as a rule-based classifier with
explicit confidence/method tags on every result.

| Source | What it gives us | Caveats |
|---|---|---|
| **World Port Index (WPI)** | Global list of ports with one lat/lon per port, harbor type/size/use | One point per *port area*, not per terminal/berth. Free CSV from NGA MSI. |
| **Global Energy Monitor (GEM) - Oil & Gas Infrastructure Tracker** | Terminal-level points: type (import/export/storage), status, capacity -> **seed labels** | Requires free account to download. Coverage best for major terminals; smaller/older terminals may be missing - fill gaps with `manual_terminal_labels.csv`. |
| **OpenStreetMap (Overpass API)** | Berth/jetty/SBM/mooring geometry, can refine WPI's port-level point to terminal-level | Patchy, especially offshore. Treat as supplementary/refinement, not primary. Doesn't contribute seed labels. |
| **Your AIS feed (AISstream)** | Vessel stop locations/durations (via SOG + NavigationalStatus), voyage ordering per vessel, tanker pairing for STS | Does **NOT** give per-voyage draught. Need an **absolute capture timestamp** per message (the AIS internal `Timestamp` field is just seconds-of-minute, not usable for duration calculations). |

## Methodology: terminal-seed + voyage-sequence alternation

For each port call by a tanker, we want one of:

- **LOAD** - vessel loaded cargo here
- **DISCHARGE** - vessel discharged cargo here
- **BOTH** - terminal genuinely does both (e.g. large refinery complex,
  storage/trading hub) - can't be resolved to one direction per call
- **STS** - this "stop" is at a known ship-to-ship transfer area, not a
  fixed berth
- **UNCERTAIN** - no evidence anywhere in this vessel's recorded
  itinerary to anchor the alternation chain

### Step 1: stop detection (speed + navigational status)
A "stop" is a contiguous period where a vessel is stationary, defined as
`SOG <= 0.5 kn` **OR** `NavigationalStatus` indicates AT ANCHOR (1) or
MOORED (5) - whichever signal is available. NavigationalStatus is the
stronger signal where present (it's set by the crew, not inferred from
GPS noise). Each stop is tagged `MOORED`, `ANCHORED`, or `DRIFTING`
(SOG-only, lower confidence - also a candidate for STS). A stop also
splits if there's a >6h gap in position reports (`max_gap_hours`),
preventing two separate port calls from merging into one.

### Step 2: terminal seed labels (the "ground truth" anchor)
Each terminal in the master reference table gets a `seed_label` -
`LOAD` / `DISCHARGE` / `BOTH` / `STS_HUB` / `UNKNOWN` - derived from:
  - **GEM terminal_type/status keywords** (rough heuristic: "export" ->
    LOAD, "import"/"discharge" -> DISCHARGE, "refinery"/"storage"/"hub"
    -> BOTH, "STS"/"transshipment"/"lightering" -> STS_HUB)
  - **`data/raw/manual_terminal_labels.csv`** (HIGHLY RECOMMENDED - a
    simple `match,label` CSV where `match` is a substring of the
    terminal name/id. This is the highest-value file you can create:
    labelling even 5-10 major terminals you already know the role of
    seeds the alternation chain for a large share of voyages. See
    `data/raw/manual_terminal_labels.csv.example`.)

### Step 3: voyage-sequence alternation (the key step)
For each vessel, walk its port calls in chronological order:
  - Calls at terminals with `seed_label` LOAD/DISCHARGE get that label
    directly (`inference_method = terminal_seed`) and anchor the chain.
  - Calls at BOTH/STS_HUB terminals get `BOTH`/`STS` and pass through
    without breaking the chain (we don't know which direction they'd
    push it).
  - Calls at UNKNOWN terminals get the *opposite* of the nearest
    anchored call before or after them in the sequence
    (`inference_method = sequence_alternation`).
  - If a vessel's ENTIRE recorded itinerary has no anchored call at all,
    those calls stay `UNCERTAIN` (`inference_method = unresolved`).

Two consecutive calls that BOTH have a direct `terminal_seed` and are
the SAME direction (two LOADs or two DISCHARGEs in a row) get
`sequence_flag = consecutive_load/consecutive_discharge` - a real
pattern in multi-port part-cargo operations, not necessarily an error,
but worth a look.

### Step 4: cross-vessel terminal-majority pass
Once Step 3 has run for all vessels, some terminals will have several
confidently-classified visits even though they started as UNKNOWN (their
visits were filled via alternation). If a terminal has >= 3 classified
visits and >= 70% agree on LOAD or DISCHARGE, any remaining `UNCERTAIN`
visits AT THAT TERMINAL get relabelled to that majority direction
(`inference_method = terminal_majority`). This is a separate,
lower-confidence inference path - "most vessels load here, so this one
probably did too" rather than voyage-specific evidence.

### Step 5: port-level rollup
Each terminal gets a `port_label` (LOAD / DISCHARGE / BOTH / STS_HUB /
INSUFFICIENT_DATA) and a `confidence` (share of classified visits
agreeing with the label), based on `inferred_classification` across all
visits.

## The plan (3 phases)

### Phase 1 - Reference data (static)
1. Download WPI CSV -> `data/raw/`
2. Download GEM Oil & Gas Infrastructure Tracker -> `data/raw/`
3. (Optional) Query OSM via Overpass for a region of interest
4. (Recommended) Create `data/raw/manual_terminal_labels.csv` - even a
   handful of known major terminals (LOAD/DISCHARGE/BOTH/STS_HUB) goes a
   long way - see `.example` file.
5. Run `clean_ports.py` -> spatially match GEM/OSM points to nearest WPI
   port (within 15 km, configurable), derive `seed_label` from GEM
   keywords + manual overrides -> `data/processed/master_terminals.parquet`

   This gives you a **terminal-level geofence + seed-label reference
   table** - the thing everything else gets joined against.

### Phase 2 - AIS-derived stops & voyages
1. Export your AIS PositionReport + ShipStaticData streams to
   `data/raw/` as parquet (recommended) or newline-delimited JSON,
   **including an absolute timestamp column**.
2. `detect_stationary_periods()` - find "stops" using SOG + NavigationalStatus
3. `assign_terminal()` - geofence each stop against the master terminal
   table (within 5 km, configurable)
4. `attach_terminal_seed()` - attach each stop's terminal `seed_label`

### Phase 3 - Voyage-sequence inference, rollup & STS
1. `apply_voyage_sequence_inference()` - per-vessel, alternate
   LOAD<->DISCHARGE through the voyage, anchored by seeded terminals
2. `apply_terminal_majority_pass()` - fill remaining UNCERTAIN visits
   using their terminal's now-confident majority label
3. `aggregate_port_classification()` - per terminal: LOAD / DISCHARGE /
   BOTH / STS_HUB / INSUFFICIENT_DATA + a confidence score
4. `detect_sts_events()` - find tanker pairs stationary & close together
   (<0.5 km, >2h) **outside** any terminal geofence -> candidate STS
   events for manual review

## Project structure

```
oil_port_classifier/
├── data/
│   ├── raw/            # downloaded WPI/GEM, OSM cache, AIS exports,
│   │                    # manual_terminal_labels.csv (gitignored)
│   └── processed/      # master_terminals.parquet etc.
├── src/
│   ├── load_data.py     # readers for WPI, GEM, OSM, AIS
│   ├── clean_ports.py    # spatial matching + seed-label derivation -> master terminal table
│   ├── geo_utils.py       # haversine, KD-tree nearest-neighbour (no GDAL)
│   ├── classify.py        # stop detection, voyage-sequence inference, STS, rollup
│   └── run_pipeline.py     # orchestrates everything
├── outputs/
│   ├── stops_classified.csv     # per-visit detail (incl. inference_method)
│   ├── port_classification.csv  # per-terminal LOAD/DISCHARGE/BOTH/STS_HUB + confidence
│   └── sts_candidates.csv       # candidate STS pairs/locations
├── requirements.txt
└── .gitignore
```

## Setup

```bash
cd oil_port_classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Getting the data

1. **WPI**: https://msi.nga.mil/Publications/WPI -> download the
   **Shapefile** export (not the MS Access database - avoids needing
   Access/ODBC drivers; not the PDF - not structured data). Drop the
   downloaded `.zip` (or extracted `.shp`/`.shx`/`.dbf`) straight into
   `data/raw/` with "wpi" in the filename - `load_wpi()` reads shapefiles
   directly via `pyshp` (pure Python, no GDAL needed) and also accepts a
   plain CSV if NGA's site offers one in future.
2. **GEM**: https://globalenergymonitor.org/projects/global-oil-infrastructure-tracker/
   -> free signup -> download xlsx -> save to `data/raw/` with "gem" in
   the filename.
3. **OSM**: handled automatically via Overpass if you pass a `bbox` to
   `main()` in `run_pipeline.py`. Keep bboxes regional (e.g. one
   country/strait at a time) - the public Overpass instance throttles
   large global queries.
4. **AIS**: export your DB's PositionReport and ShipStaticData to
   `data/raw/` as `ais_positions.parquet` and `ais_static.parquet`
   (or `.jsonl`). **Make sure there's an absolute timestamp column**
   (default expected name: `MsgTimestamp`).
5. **Manual terminal labels** (recommended): copy
   `data/raw/manual_terminal_labels.csv.example` to
   `data/raw/manual_terminal_labels.csv` and fill in any terminals you
   already know the role of - `match,label` where `label` is
   LOAD / DISCHARGE / BOTH / STS_HUB. `match` is matched as a substring
   against terminal name/id (case-insensitive).

## Running

```bash
python src/run_pipeline.py
```

Edit the `main()` call at the bottom of `run_pipeline.py` to pass an
OSM bbox (`(south, west, north, east)`) and your AIS timestamp column
name if different from `MsgTimestamp`.

## Tuning knobs (top of `classify.py`)

| Constant | Default | Meaning |
|---|---|---|
| `SOG_STATIONARY_THRESHOLD` | 0.5 kn | Below this = "not moving" |
| `MIN_STOP_DURATION_MIN` | 120 min | Minimum stop length to count as a port call |
| `TERMINAL_RADIUS_KM` | 5 km | Geofence radius around a terminal |
| `max_gap_hours` (detect_stationary_periods) | 6 h | Split a stop if position reports gap exceeds this |
| `STS_DISTANCE_KM` | 0.5 km | Max distance between two vessels for an STS candidate |
| `STS_MIN_DURATION_MIN` | 120 min | Minimum duration of proximity for STS candidate |
| `TERMINAL_MAJORITY_MIN_CONFIDENCE` | 0.7 | Min agreement for Step 4's cross-vessel pass |
| `TERMINAL_MAJORITY_MIN_VISITS` | 3 | Min classified visits before Step 4 trusts a terminal |

## Known limitations / next steps

- **Coverage depends on seed labels.** A vessel whose ENTIRE recorded
  itinerary touches only UNKNOWN terminals stays `UNCERTAIN` until either
  (a) you add a manual label for one of those terminals, or (b) the
  terminal-majority pass eventually resolves it once enough *other*
  vessels' voyages have anchored it. The single highest-leverage action
  is filling out `manual_terminal_labels.csv` for the busiest terminals
  in your AIS coverage area.
- **GEM keyword matching is rough.** `classify_terminal_seed()` in
  `clean_ports.py` uses simple regex keyword matches on GEM's
  terminal_type/status/name text - GEM's taxonomy varies across exports,
  so review `terminals["seed_label"]` for your region and correct via
  manual overrides where it looks wrong.
- **`BOTH`-seeded terminals (refineries, storage hubs) don't anchor the
  alternation chain** by design - we genuinely don't know which
  direction they'd push it. If you have finer-grained knowledge (e.g.
  "this refinery's tanker jetty is crude-in only"), label it
  LOAD/DISCHARGE specifically rather than BOTH.
- **`sequence_flag` (consecutive same-direction calls)** is informational,
  not necessarily wrong - multi-port part-cargo loading/discharging is
  common in the crude trade. Worth spot-checking a sample.
- STS detection is a **first-pass filter** - review `sts_candidates.csv`
  manually before treating it as ground truth. Known false positives:
  vessels rafted at anchorage near (but technically outside) a port
  geofence.
- No ML in this version by design - get the rule-based baseline working
  and validated against a handful of known ports first, then consider
  a classifier trained on the resulting labelled dataset if you want to
  generalise further.
- If you ever ingest a different AIS feed that DOES report dynamic
  per-voyage draught, `classify.py`'s LEGACY section
  (`attach_draught_change` / `classify_visit_by_draught`) gives you a
  second, independent signal you could combine with the seed/alternation
  approach for higher confidence.
