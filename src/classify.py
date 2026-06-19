"""
classify.py
------------
Core heuristics turning raw AIS PositionReport / ShipStaticData streams
into port-/terminal-level LOAD / DISCHARGE / BOTH / STS classifications.

IMPORTANT - timestamps:
  The AIS "Timestamp" field in PositionReport is the AIS-internal
  second-of-minute (0-59) and is NOT a usable absolute time. Your DB
  export needs a real capture/received timestamp column (e.g.
  "MsgTimestamp", epoch seconds or ISO string). All functions below take
  a `time_col` parameter - point it at whatever your export calls that
  field.

IMPORTANT - draught:
  AISstream's ShipStaticData.MaximumStaticDraught is a STATIC HULL
  CONSTANT (the vessel's configured maximum draught), NOT a per-voyage
  reading. It does NOT change between a laden and ballast condition, so
  it CANNOT be used to detect "did this vessel get heavier/lighter at
  this port?". The draught-delta approach from earlier versions of this
  module is therefore not usable with AISstream and has been moved to
  the LEGACY section at the bottom of this file (kept only in case you
  ever ingest a feed that *does* report dynamic draught).

Pipeline (see also README.md):
  1. detect_stationary_periods()       -> per-vessel "stops" (SOG + NavigationalStatus)
  2. assign_terminal()                 -> attach nearest terminal (geofence)
  3. attach_terminal_seed()            -> attach each terminal's known role
                                           (seed_label, from clean_ports.py:
                                           GEM keyword match + manual overrides)
  4. apply_voyage_sequence_inference() -> per-VESSEL, alternate LOAD<->DISCHARGE
                                           through the voyage, seeded by
                                           terminals with a known seed_label
  5. apply_terminal_majority_pass()    -> fill any remaining UNCERTAIN visits
                                           using the now-confident label of
                                           their terminal (cross-vessel)
  6. aggregate_port_classification()   -> per-terminal rollup + confidence
  7. detect_sts_events()               -> vessel-pair STS candidates

Core idea (the actual goal of this module):
  A tanker's cargo state alternates: LOAD (becomes laden) -> sail laden
  -> DISCHARGE (becomes ballast) -> sail ballast -> LOAD -> ...
  Without per-voyage draught, we can't observe this directly - but if we
  know the ROLE of even a few terminals in a vessel's itinerary (from
  GEM data or your own domain knowledge via manual_terminal_labels.csv),
  the alternation rule fills in the rest of that vessel's calls. Step 5
  then lets terminals that became confident through step 4 (across many
  vessels) "vote" on any calls still left UNCERTAIN.

Vessel filter:
  AIS ship-type codes 80-89 = "Tanker" (incl. 80=generic tanker,
  83=tanker carrying dangerous goods cat C, etc). We filter to 80-89 by
  default for "is this a tanker" checks; widen TANKER_TYPES if you also
  want e.g. combination carriers or offshore supply vessels in STS
  detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from geo_utils import haversine_km, nearest_neighbours

TANKER_TYPES = set(range(80, 90))  # AIS ship-type codes 80-89

# AIS NavigationalStatus codes (ITU-R M.1371) relevant to "is this vessel
# parked somewhere doing cargo ops, or just slow in transit?"
NAV_STATUS_AT_ANCHOR = 1
NAV_STATUS_MOORED = 5
STATIONARY_NAV_STATUSES = {NAV_STATUS_AT_ANCHOR, NAV_STATUS_MOORED}

# Thresholds - tune these against your own data / known port calls.
SOG_STATIONARY_THRESHOLD = 0.5      # knots
MIN_STOP_DURATION_MIN = 120         # minutes - below this, treat as transit/drift
TERMINAL_RADIUS_KM = 5.0            # geofence radius around a terminal point
STS_DISTANCE_KM = 0.5               # vessels within this distance = candidate STS pair
STS_MIN_DURATION_MIN = 120          # both stationary & close for at least this long
STS_EXCLUDE_TERMINAL_RADIUS_KM = 2.0  # ignore pairs inside a terminal geofence (that's just berthing)

# Stage 5 (cross-vessel terminal-majority pass) thresholds
TERMINAL_MAJORITY_MIN_CONFIDENCE = 0.7
TERMINAL_MAJORITY_MIN_VISITS = 3


# ---------------------------------------------------------------------------
# 1. Stationary period ("stop") detection
# ---------------------------------------------------------------------------

def detect_stationary_periods(
    positions: pd.DataFrame,
    time_col: str = "MsgTimestamp",
    sog_threshold: float = SOG_STATIONARY_THRESHOLD,
    min_duration_min: float = MIN_STOP_DURATION_MIN,
    use_nav_status: bool = True,
    max_gap_hours: float = 6.0,
) -> pd.DataFrame:
    """
    For each vessel (UserID), find contiguous time windows where the vessel
    is "stationary", lasting at least `min_duration_min` minutes.

    "Stationary" = SOG <= sog_threshold, OR (if use_nav_status and the
    column is present) NavigationalStatus indicates AT ANCHOR (1) or
    MOORED (5). NavigationalStatus is the more reliable signal where
    available - a vessel can show SOG~0 while drifting, but status 5
    ("moored") is set by the crew and means it's actually alongside.

    Returns one row per stop:
      UserID, start_time, end_time, duration_min,
      mean_lat, mean_lon, n_points, stop_type
    where stop_type is one of:
      "MOORED"   - majority of points report NavigationalStatus == 5
      "ANCHORED" - majority report NavigationalStatus == 1
      "DRIFTING" - stationary by SOG only, no mooring/anchor status seen
                   (worth treating with lower confidence - could be
                   adverse current, waiting, or an STS candidate)

    `max_gap_hours` splits an otherwise-contiguous stationary block if
    consecutive position reports are more than this far apart in time -
    this prevents two separate port calls (with a signal gap / transit
    in between, but no intervening "moving" reports) from being merged
    into one stop.
    """
    df = positions.copy()
    if time_col not in df.columns:
        raise KeyError(
            f"'{time_col}' not found in positions dataframe. "
            f"Available columns: {list(df.columns)}. "
            "You need an absolute capture timestamp - see module docstring."
        )

    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col, "Sog", "Latitude", "Longitude"])
    df = df.sort_values(["UserID", time_col])

    has_nav = use_nav_status and "NavigationalStatus" in df.columns
    sog_stationary = df["Sog"] <= sog_threshold
    if has_nav:
        nav_stationary = df["NavigationalStatus"].isin(STATIONARY_NAV_STATUSES)
        df["is_stationary"] = sog_stationary | nav_stationary
    else:
        df["is_stationary"] = sog_stationary

    stops = []
    for user_id, g in df.groupby("UserID", sort=False):
        g = g.reset_index(drop=True)
        # group consecutive rows with the same is_stationary value, AND
        # split on time gaps larger than max_gap_hours
        gap_hours = g[time_col].diff().dt.total_seconds() / 3600.0
        new_block = (g["is_stationary"] != g["is_stationary"].shift()) | (gap_hours > max_gap_hours)
        block_id = new_block.cumsum()
        for _, block in g.groupby(block_id):
            if not block["is_stationary"].iloc[0]:
                continue
            start, end = block[time_col].iloc[0], block[time_col].iloc[-1]
            duration_min = (end - start).total_seconds() / 60.0
            if duration_min < min_duration_min:
                continue

            stop_type = "DRIFTING"
            if has_nav:
                status_counts = block["NavigationalStatus"].value_counts()
                if status_counts.get(NAV_STATUS_MOORED, 0) >= status_counts.get(NAV_STATUS_AT_ANCHOR, 0) \
                        and status_counts.get(NAV_STATUS_MOORED, 0) > 0:
                    stop_type = "MOORED"
                elif status_counts.get(NAV_STATUS_AT_ANCHOR, 0) > 0:
                    stop_type = "ANCHORED"

            stops.append({
                "UserID": user_id,
                "start_time": start,
                "end_time": end,
                "duration_min": duration_min,
                "mean_lat": block["Latitude"].mean(),
                "mean_lon": block["Longitude"].mean(),
                "n_points": len(block),
                "stop_type": stop_type,
            })

    return pd.DataFrame(stops)


# ---------------------------------------------------------------------------
# 2. Assign each stop to a terminal (geofence)
# ---------------------------------------------------------------------------

def assign_terminal(
    stops: pd.DataFrame,
    terminals: pd.DataFrame,
    radius_km: float = TERMINAL_RADIUS_KM,
) -> pd.DataFrame:
    """
    Attach the nearest terminal to each stop, dropping stops further than
    `radius_km` from any terminal (these are likely anchorages, drifting,
    or open-sea stops - candidates for STS, handled separately).
    """
    if not len(stops):
        return stops.assign(terminal_id=pd.Series(dtype=object),
                             terminal_name=pd.Series(dtype=object),
                             dist_to_terminal_km=pd.Series(dtype=float))

    stops_for_match = stops.rename(columns={"mean_lat": "latitude", "mean_lon": "longitude"})
    matched = nearest_neighbours(stops_for_match, terminals, k=1)

    out = stops.copy()
    out["dist_to_terminal_km"] = matched["nearest_dist_km_0"]
    within = out["dist_to_terminal_km"] <= radius_km

    terminal_ids = terminals["terminal_id"].to_numpy()
    terminal_names = terminals["terminal_name"].to_numpy()
    idx = matched["nearest_idx_0"].to_numpy()

    out["terminal_id"] = np.where(within, terminal_ids[idx], None)
    out["terminal_name"] = np.where(within, terminal_names[idx], None)
    return out


# ---------------------------------------------------------------------------
# 3. Attach each terminal's known role (seed label)
# ---------------------------------------------------------------------------

def attach_terminal_seed(stops: pd.DataFrame, terminals: pd.DataFrame) -> pd.DataFrame:
    """
    Attach `terminal_seed_label` (LOAD / DISCHARGE / BOTH / STS_HUB / UNKNOWN)
    from the master terminal table (clean_ports.build_master_terminal_table,
    which derives it from GEM keywords + manual_terminal_labels.csv).

    Stops with no terminal_id (terminal_id is None - open-sea/anchorage
    stops) get terminal_seed_label = "UNKNOWN".
    """
    out = stops.copy()
    if not len(out):
        out["terminal_seed_label"] = pd.Series(dtype=object)
        return out

    seed_map = terminals.set_index("terminal_id")["seed_label"].to_dict()
    out["terminal_seed_label"] = out["terminal_id"].map(seed_map).fillna("UNKNOWN")
    return out


# ---------------------------------------------------------------------------
# 4. Voyage-sequence inference (laden <-> ballast alternation)
# ---------------------------------------------------------------------------
#
# For a single tanker, cargo operations alternate: LOAD -> sail laden ->
# DISCHARGE -> sail ballast -> LOAD -> ... Given each call's
# terminal_seed_label (LOAD / DISCHARGE / BOTH / STS_HUB / UNKNOWN):
#
#   - LOAD / DISCHARGE seeds are direct evidence for that call, and seed
#     the alternation chain (forward AND backward).
#   - BOTH / STS_HUB seeds don't tell us which direction the chain should
#     continue in, so they pass through without breaking the chain - they
#     get their own inferred_classification ("BOTH" / "STS") but don't
#     update last_known/next_known.
#   - UNKNOWN seeds get filled from the nearest known neighbour in the
#     chain (forward pass first, then a backward pass for any leading
#     UNKNOWNs). If NOTHING in the vessel's whole sequence has a known
#     seed, those calls end up "UNCERTAIN".

_OPPOSITE = {"LOAD": "DISCHARGE", "DISCHARGE": "LOAD"}


def infer_voyage_sequence(vessel_stops: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the alternation rule across one vessel's port calls, ordered by
    time. `vessel_stops` must have a 'terminal_seed_label' column and be
    for a SINGLE UserID. Adds:

      inferred_classification : LOAD | DISCHARGE | BOTH | STS | UNCERTAIN
      inference_method : "terminal_seed" | "terminal_seed_both" |
                          "terminal_seed_sts" | "sequence_alternation" |
                          "unresolved"
      sequence_flag : "" | "consecutive_load" | "consecutive_discharge"
                       (flags runs of >=2 same-type calls that BOTH have
                       a direct terminal_seed - i.e. genuinely two LOADs
                       or two DISCHARGEs in a row according to known
                       terminal roles. Common in multi-port part-cargo
                       operations; worth a glance.)
    """
    v = vessel_stops.sort_values("start_time").reset_index(drop=True)
    n = len(v)
    inferred: list[str | None] = [None] * n
    method: list[str | None] = [None] * n

    for i, seed in enumerate(v["terminal_seed_label"]):
        if seed in ("LOAD", "DISCHARGE"):
            inferred[i] = seed
            method[i] = "terminal_seed"
        elif seed == "BOTH":
            inferred[i] = "BOTH"
            method[i] = "terminal_seed_both"
        elif seed == "STS_HUB":
            inferred[i] = "STS"
            method[i] = "terminal_seed_sts"
        # else seed == "UNKNOWN" -> leave None for now

    # Forward pass: propagate alternation from earlier known calls forward.
    last_known = None
    for i in range(n):
        if inferred[i] in ("LOAD", "DISCHARGE"):
            last_known = inferred[i]
        elif inferred[i] is None and last_known is not None:
            inferred[i] = _OPPOSITE[last_known]
            method[i] = "sequence_alternation"
            last_known = inferred[i]
        # BOTH / STS pass through without updating last_known

    # Backward pass: resolve any leading Nones using the first known call
    # after them.
    next_known = None
    for i in range(n - 1, -1, -1):
        if inferred[i] in ("LOAD", "DISCHARGE"):
            next_known = inferred[i]
        elif inferred[i] is None:
            if next_known is not None:
                inferred[i] = _OPPOSITE[next_known]
                method[i] = "sequence_alternation"
                next_known = inferred[i]
            else:
                inferred[i] = "UNCERTAIN"
                method[i] = "unresolved"
        # BOTH / STS pass through without updating next_known

    # Flag consecutive same-direction calls that BOTH have a direct
    # terminal_seed (not sequence-filled).
    flags = [""] * n
    for i in range(1, n):
        if (method[i] == "terminal_seed" and method[i - 1] == "terminal_seed"
                and inferred[i] == inferred[i - 1]):
            tag = "consecutive_load" if inferred[i] == "LOAD" else "consecutive_discharge"
            flags[i - 1] = tag
            flags[i] = tag

    out = v.copy()
    out["inferred_classification"] = inferred
    out["inference_method"] = method
    out["sequence_flag"] = flags
    return out


def apply_voyage_sequence_inference(stops_with_seed: pd.DataFrame) -> pd.DataFrame:
    """Apply infer_voyage_sequence() per-vessel across the whole stops table."""
    if not len(stops_with_seed):
        return stops_with_seed.assign(
            inferred_classification=pd.Series(dtype=object),
            inference_method=pd.Series(dtype=object),
            sequence_flag=pd.Series(dtype=object),
        )
    parts = [infer_voyage_sequence(g) for _, g in stops_with_seed.groupby("UserID", sort=False)]
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# 5. Cross-vessel terminal-majority pass
# ---------------------------------------------------------------------------

def apply_terminal_majority_pass(
    stops: pd.DataFrame,
    min_confidence: float = TERMINAL_MAJORITY_MIN_CONFIDENCE,
    min_visits: int = TERMINAL_MAJORITY_MIN_VISITS,
) -> pd.DataFrame:
    """
    After per-vessel voyage-sequence inference, some calls remain
    "UNCERTAIN" - typically a vessel whose ENTIRE recorded itinerary
    touches only terminals with unknown seed labels.

    This pass looks at the (now mostly-filled) per-terminal rollup: if a
    terminal has >= `min_visits` classified calls and its port_label
    (LOAD or DISCHARGE) has confidence >= `min_confidence`, any
    "UNCERTAIN" calls AT THAT TERMINAL get relabelled to that port_label
    with inference_method="terminal_majority".

    This is a SEPARATE, lower-confidence inference path - it's "most
    vessels load here, so this one probably did too", not voyage-specific
    evidence. Keep an eye on `inference_method` if you need to distinguish.
    """
    if not len(stops) or "inferred_classification" not in stops.columns:
        return stops

    port_class = aggregate_port_classification(stops)
    strong = port_class[
        port_class["port_label"].isin(["LOAD", "DISCHARGE"])
        & (port_class["confidence"] >= min_confidence)
        & (port_class["n_visits"] >= min_visits)
    ]
    label_map = dict(zip(strong["terminal_id"], strong["port_label"]))
    if not label_map:
        return stops

    out = stops.copy()
    mask = (out["inferred_classification"] == "UNCERTAIN") & out["terminal_id"].isin(label_map)
    out.loc[mask, "inferred_classification"] = out.loc[mask, "terminal_id"].map(label_map)
    out.loc[mask, "inference_method"] = "terminal_majority"
    return out


# ---------------------------------------------------------------------------
# 6. Port-level rollup
# ---------------------------------------------------------------------------

def aggregate_port_classification(
    classified_visits: pd.DataFrame,
    classification_col: str = "inferred_classification",
) -> pd.DataFrame:
    """
    Roll per-visit classifications up to a per-terminal label + confidence.

    Output columns:
      terminal_id, terminal_name, n_visits,
      n_load, n_discharge, n_both, n_sts, n_uncertain,
      port_label  in {LOAD, DISCHARGE, BOTH, STS_HUB, INSUFFICIENT_DATA}
      confidence  (0-1, share of classified visits agreeing with port_label;
                   for STS_HUB/BOTH, share of total visits)
    """
    df = classified_visits.dropna(subset=["terminal_id"])
    if not len(df):
        return pd.DataFrame(columns=[
            "terminal_id", "terminal_name", "n_visits", "n_load",
            "n_discharge", "n_both", "n_sts", "n_uncertain", "port_label", "confidence",
        ])

    rows = []
    for (tid, tname), g in df.groupby(["terminal_id", "terminal_name"], dropna=False):
        counts = g[classification_col].value_counts()
        n_load = int(counts.get("LOAD", 0))
        n_disc = int(counts.get("DISCHARGE", 0))
        n_both = int(counts.get("BOTH", 0))
        n_sts = int(counts.get("STS", 0))
        n_unc = int(counts.get("UNCERTAIN", 0))
        n_total = len(g)
        n_classified = n_load + n_disc

        if n_sts > n_total / 2:
            label, conf = "STS_HUB", n_sts / n_total
        elif n_classified == 0 and n_both > 0:
            label, conf = "BOTH", n_both / n_total
        elif n_classified == 0:
            label, conf = "INSUFFICIENT_DATA", 0.0
        elif n_load > 0 and n_disc > 0:
            ratio = min(n_load, n_disc) / max(n_load, n_disc)
            if ratio >= 0.25:
                label, conf = "BOTH", max(n_load, n_disc) / n_classified
            else:
                label = "LOAD" if n_load > n_disc else "DISCHARGE"
                conf = max(n_load, n_disc) / n_classified
        elif n_load > 0:
            label, conf = "LOAD", n_load / n_classified
        else:
            label, conf = "DISCHARGE", n_disc / n_classified

        rows.append({
            "terminal_id": tid,
            "terminal_name": tname,
            "n_visits": n_total,
            "n_load": n_load,
            "n_discharge": n_disc,
            "n_both": n_both,
            "n_sts": n_sts,
            "n_uncertain": n_unc,
            "port_label": label,
            "confidence": round(conf, 2),
        })

    return pd.DataFrame(rows).sort_values("n_visits", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. STS (ship-to-ship transfer) candidate detection
# ---------------------------------------------------------------------------

@dataclass
class STSEvent:
    user_id_a: int
    user_id_b: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_min: float
    mean_lat: float
    mean_lon: float


def detect_sts_events(
    positions: pd.DataFrame,
    terminals: pd.DataFrame,
    static_data: pd.DataFrame | None = None,
    time_col: str = "MsgTimestamp",
    sog_threshold: float = SOG_STATIONARY_THRESHOLD,
    distance_km: float = STS_DISTANCE_KM,
    min_duration_min: float = STS_MIN_DURATION_MIN,
    exclude_terminal_radius_km: float = STS_EXCLUDE_TERMINAL_RADIUS_KM,
    time_bin: str = "30min",
) -> pd.DataFrame:
    """
    Heuristic STS detection:

      1. Optionally restrict to tanker MMSIs (via static_data Type 80-89).
      2. Keep only stationary positions (SOG <= sog_threshold).
      3. Drop positions inside any terminal geofence (that's berthing,
         not STS - STS happens at sea / designated STS zones).
      4. Bin remaining positions into `time_bin` windows.
      5. Within each time bin, find vessel pairs within `distance_km`
         of each other.
      6. Keep pairs that co-occur (stationary + close) across enough
         consecutive bins to total >= `min_duration_min`.

    Returns a dataframe of candidate STS events - review manually, this
    is a first-pass filter, not a final answer (anchorage rafting near
    a port edge can also trigger this; the terminal exclusion radius
    helps but won't catch every case).
    """
    df = positions.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col, "Sog", "Latitude", "Longitude", "UserID"])
    df = df[df["Sog"] <= sog_threshold]

    if static_data is not None and "Type" in static_data.columns:
        tanker_ids = set(static_data.loc[static_data["Type"].isin(TANKER_TYPES), "UserID"])
        if tanker_ids:
            df = df[df["UserID"].isin(tanker_ids)]

    if not len(df):
        return pd.DataFrame(columns=[
            "user_id_a", "user_id_b", "start_time", "end_time", "duration_min", "mean_lat", "mean_lon",
        ])

    # Drop points inside any terminal geofence
    if len(terminals):
        matched = nearest_neighbours(
            df.rename(columns={"Latitude": "latitude", "Longitude": "longitude"}),
            terminals, k=1,
        )
        df = df[matched["nearest_dist_km_0"].to_numpy() > exclude_terminal_radius_km]

    if not len(df):
        return pd.DataFrame(columns=[
            "user_id_a", "user_id_b", "start_time", "end_time", "duration_min", "mean_lat", "mean_lon",
        ])

    df["time_bin"] = df[time_col].dt.floor(time_bin)

    # For each time bin, find close pairs
    pair_bins: dict[tuple[int, int], list[pd.Timestamp]] = {}
    pair_pos: dict[tuple[int, int], list[tuple[float, float]]] = {}

    for bin_ts, g in df.groupby("time_bin"):
        g = g.drop_duplicates(subset="UserID")  # one position per vessel per bin
        if len(g) < 2:
            continue
        users = g["UserID"].to_numpy()
        lats = g["Latitude"].to_numpy()
        lons = g["Longitude"].to_numpy()
        n = len(g)
        for i in range(n):
            for j in range(i + 1, n):
                d = haversine_km(lats[i], lons[i], lats[j], lons[j])
                if d <= distance_km:
                    key = (min(users[i], users[j]), max(users[i], users[j]))
                    pair_bins.setdefault(key, []).append(bin_ts)
                    pair_pos.setdefault(key, []).append(((lats[i] + lats[j]) / 2, (lons[i] + lons[j]) / 2))

    bin_minutes = pd.Timedelta(time_bin).total_seconds() / 60.0
    events = []
    for (ua, ub), bins in pair_bins.items():
        bins = sorted(bins)
        positions_avg = pair_pos[(ua, ub)]
        # group consecutive bins
        run_start = bins[0]
        prev = bins[0]
        run_positions = [positions_avg[0]]
        for ts, pos in zip(bins[1:], positions_avg[1:]):
            if (ts - prev) <= pd.Timedelta(time_bin) * 1.5:
                run_positions.append(pos)
                prev = ts
                continue
            duration = (prev - run_start).total_seconds() / 60.0 + bin_minutes
            if duration >= min_duration_min:
                lat = np.mean([p[0] for p in run_positions])
                lon = np.mean([p[1] for p in run_positions])
                events.append(STSEvent(ua, ub, run_start, prev, duration, lat, lon))
            run_start = ts
            prev = ts
            run_positions = [pos]
        duration = (prev - run_start).total_seconds() / 60.0 + bin_minutes
        if duration >= min_duration_min:
            lat = np.mean([p[0] for p in run_positions])
            lon = np.mean([p[1] for p in run_positions])
            events.append(STSEvent(ua, ub, run_start, prev, duration, lat, lon))

    return pd.DataFrame([e.__dict__ for e in events])


# ---------------------------------------------------------------------------
# LEGACY - draught-delta classification
# ---------------------------------------------------------------------------
#
# NOT USABLE WITH AISSTREAM. Kept only for users of a different AIS feed
# that genuinely reports per-voyage draught (some commercial AIS
# providers parse a "current draught" separately from the static hull
# max). If your feed only gives MaximumStaticDraught (constant per
# vessel), draught_data_ok will be effectively useless - every call's
# delta will be ~0 and everything collapses to NO_CHANGE/UNKNOWN.
# Use the seed/voyage-sequence pipeline above instead.

DRAUGHT_CHANGE_THRESHOLD_M = 0.5  # metres


def attach_draught_change(
    stops: pd.DataFrame,
    static_data: pd.DataFrame,
    time_col: str = "MsgTimestamp",
    max_gap_hours: float = 96.0,
) -> pd.DataFrame:
    """[LEGACY] See module note above - requires a feed with dynamic draught."""
    sd = static_data.copy()
    if time_col not in sd.columns:
        raise KeyError(f"'{time_col}' not found in static_data columns: {list(sd.columns)}")
    sd[time_col] = pd.to_datetime(sd[time_col], utc=True, errors="coerce")
    sd = sd.dropna(subset=[time_col, "MaximumStaticDraught", "UserID"])
    sd = sd.sort_values(["UserID", time_col])

    before_vals, after_vals = [], []
    before_gaps, after_gaps = [], []
    for _, stop in stops.iterrows():
        vessel = sd[sd["UserID"] == stop["UserID"]]
        if vessel.empty:
            before_vals.append(np.nan)
            after_vals.append(np.nan)
            before_gaps.append(np.nan)
            after_gaps.append(np.nan)
            continue

        before_candidates = vessel[vessel[time_col] <= stop["start_time"]]
        after_candidates = vessel[vessel[time_col] >= stop["end_time"]]

        if len(before_candidates):
            before_vals.append(before_candidates["MaximumStaticDraught"].iloc[-1])
            before_gaps.append((stop["start_time"] - before_candidates[time_col].iloc[-1]).total_seconds() / 3600.0)
        else:
            before_vals.append(np.nan)
            before_gaps.append(np.nan)

        if len(after_candidates):
            after_vals.append(after_candidates["MaximumStaticDraught"].iloc[0])
            after_gaps.append((after_candidates[time_col].iloc[0] - stop["end_time"]).total_seconds() / 3600.0)
        else:
            after_vals.append(np.nan)
            after_gaps.append(np.nan)

    out = stops.copy()
    out["draught_before"] = before_vals
    out["draught_after"] = after_vals
    out["draught_delta"] = out["draught_after"] - out["draught_before"]
    out["draught_before_gap_h"] = before_gaps
    out["draught_after_gap_h"] = after_gaps
    out["draught_data_ok"] = (
        out["draught_before_gap_h"].le(max_gap_hours) &
        out["draught_after_gap_h"].le(max_gap_hours)
    )
    return out


def classify_visit_by_draught(row: pd.Series, threshold_m: float = DRAUGHT_CHANGE_THRESHOLD_M) -> str:
    """[LEGACY] Classify a single port call from draught_delta. See module note above."""
    if row.get("draught_data_ok") is False:
        return "UNKNOWN"
    delta = row.get("draught_delta")
    if pd.isna(delta):
        return "UNKNOWN"
    if delta > threshold_m:
        return "LOAD"
    if delta < -threshold_m:
        return "DISCHARGE"
    return "NO_CHANGE"
