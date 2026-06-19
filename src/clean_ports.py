"""
clean_ports.py
--------------
Build a single "master terminal reference table" by combining WPI,
GEM and OSM into one dataframe with deduplicated, fuzzy-matched
locations.

Strategy:
  1. Normalise names (lowercase, strip punctuation/whitespace) for
     human-readable cross-checking (not used for matching itself -
     spatial matching is far more reliable for ports than name
     matching, since the same terminal gets wildly different names
     across datasets).
  2. Treat WPI as the spatial backbone (one row per "port area").
  3. For each GEM terminal and each OSM feature, find the nearest WPI
     port within MATCH_RADIUS_KM and attach it as a child terminal.
     Anything with no WPI match within range becomes its own
     standalone terminal row (common for offshore SBMs / FSOs that
     WPI doesn't list).
  4. Output one row per *terminal* (not per port), each carrying:
       - terminal_id (synthetic)
       - terminal_name
       - latitude / longitude  (terminal-level, e.g. GEM/OSM coords if
         available, else WPI port coords)
       - wpi_port_name / wpi_number (parent port, if matched)
       - sources (list of contributing datasets)
       - gem_terminal_type / gem_status (if from GEM)
       - osm_tags (if from OSM)
       - seed_label (LOAD / DISCHARGE / BOTH / UNKNOWN / STS_HUB)
       - seed_source ("gem_keyword" / "manual_override" / "none")

This table is the geofencing AND seed-label reference for classify.py.

----------------------------------------------------------------------
Why seed_label matters (post AISstream-draught-limitation pivot):
----------------------------------------------------------------------
AISstream's ShipStaticData.MaximumStaticDraught is a static hull
constant, NOT a per-voyage reading - so draught-delta can't tell us
LOAD vs DISCHARGE directly (see classify.py). Instead, classify.py's
voyage-sequence step needs at least SOME terminals in a vessel's
itinerary to have a known role, so it can alternate from there.
seed_label is that "known role" - derived from:
  1. GEM terminal_type/status text (keyword heuristic - rough, but
     better than nothing)
  2. Manual overrides you supply in data/raw/manual_terminal_labels.csv
     (HIGHLY RECOMMENDED - you almost certainly know offhand that e.g.
     Ras Tanura = LOAD, ARA refinery jetties = mostly DISCHARGE. A
     handful of high-traffic terminals labelled this way will seed
     alternation for a large share of voyages.)

seed_label values:
  LOAD     - vessels typically load cargo here (export terminal, FPSO
             offtake, production platform)
  DISCHARGE- vessels typically discharge cargo here (import terminal,
             refinery crude intake)
  BOTH     - genuinely does both (e.g. large refinery complex with both
             crude-in and product-out jetties, or a storage/trading hub)
  STS_HUB  - known ship-to-ship transfer area, not a fixed berth
  UNKNOWN  - no information; relies entirely on alternation from other
             terminals in the same voyage, or stays UNCERTAIN
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from geo_utils import nearest_neighbours

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# How close (km) a GEM/OSM point must be to a WPI port to be treated as
# "belonging to" that port. Oil terminals are often several km from the
# nominal port centroid (e.g. SBMs offshore), so this is intentionally
# generous. Tighten for dense port clusters.
MATCH_RADIUS_KM = 15.0


def normalise_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


# ---------------------------------------------------------------------------
# Seed label heuristics
# ---------------------------------------------------------------------------
#
# Rough keyword mapping from GEM "terminal_type"/"status"/name text to a
# LOAD/DISCHARGE/BOTH/STS_HUB seed. This is intentionally crude - GEM's
# taxonomy varies across exports/versions, and the same words can mean
# different things in different contexts. Treat this as a starting point;
# review the resulting seed_label column and correct via
# manual_terminal_labels.csv where it matters.
_LOAD_KEYWORDS = [
    r"export", r"loading", r"fpso", r"production", r"offtake", r"crude oil terminal",
]
_DISCHARGE_KEYWORDS = [
    r"import", r"discharg", r"unloading", r"receiving", r"intake",
]
_BOTH_KEYWORDS = [
    r"refiner", r"storage", r"tank farm", r"hub", r"distribution",
]
_STS_KEYWORDS = [
    r"sts", r"ship.to.ship", r"transshipment", r"trans.shipment", r"lightering",
]


def classify_terminal_seed(row: pd.Series) -> tuple[str, str]:
    """
    Derive (seed_label, seed_source) from a terminal row's GEM metadata.

    Checks gem_terminal_type, gem_status, and terminal_name (in that
    order of priority) against keyword lists. First match wins; if
    multiple categories match the same text, BOTH/STS_HUB take priority
    over a single LOAD or DISCHARGE match (since "refinery export
    terminal" should be BOTH, not LOAD).

    Returns ("UNKNOWN", "none") if nothing matches or no GEM data present.
    """
    text_parts = [
        str(row.get("gem_terminal_type") or ""),
        str(row.get("gem_status") or ""),
        str(row.get("terminal_name") or ""),
    ]
    text = " ".join(text_parts).lower()
    if not text.strip():
        return "UNKNOWN", "none"

    has_load = any(re.search(k, text) for k in _LOAD_KEYWORDS)
    has_discharge = any(re.search(k, text) for k in _DISCHARGE_KEYWORDS)
    has_both = any(re.search(k, text) for k in _BOTH_KEYWORDS)
    has_sts = any(re.search(k, text) for k in _STS_KEYWORDS)

    if has_sts:
        return "STS_HUB", "gem_keyword"
    if has_both or (has_load and has_discharge):
        return "BOTH", "gem_keyword"
    if has_load:
        return "LOAD", "gem_keyword"
    if has_discharge:
        return "DISCHARGE", "gem_keyword"
    return "UNKNOWN", "none"


def apply_manual_overrides(
    terminals: pd.DataFrame,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Apply manual seed-label overrides from a CSV with columns:
      match, label   (label in {LOAD, DISCHARGE, BOTH, STS_HUB, UNKNOWN})

    `match` is matched case-insensitively as a SUBSTRING against
    terminal_name OR terminal_id. First matching row wins per terminal.
    This is the recommended way to inject domain knowledge ("Ras Tanura
    = LOAD", "Europoort crude jetties = DISCHARGE") that GEM's keyword
    text won't reliably capture.

    If no file is found, returns `terminals` unchanged (with a printed
    note - this is not an error, just means you haven't created one yet).
    """
    if path is None:
        path = RAW_DIR / "manual_terminal_labels.csv"
    path = Path(path)
    if not path.exists():
        print(f"  (no manual override file at {path} - skipping)")
        return terminals

    overrides = pd.read_csv(path)
    overrides.columns = [c.strip().lower() for c in overrides.columns]
    if not {"match", "label"}.issubset(overrides.columns):
        raise ValueError(f"{path} must have columns 'match' and 'label'")

    out = terminals.copy()
    name_lower = out["terminal_name"].astype(str).str.lower()
    id_lower = out["terminal_id"].astype(str).str.lower()

    for _, ov in overrides.iterrows():
        pattern = str(ov["match"]).lower()
        label = str(ov["label"]).upper()
        mask = name_lower.str.contains(pattern, regex=False) | id_lower.str.contains(pattern, regex=False)
        out.loc[mask, "seed_label"] = label
        out.loc[mask, "seed_source"] = "manual_override"

    n_applied = (out["seed_source"] == "manual_override").sum()
    print(f"  applied manual overrides to {n_applied} terminal(s)")
    return out


def build_master_terminal_table(
    wpi_df: pd.DataFrame,
    gem_df: pd.DataFrame | None = None,
    osm_df: pd.DataFrame | None = None,
    match_radius_km: float = MATCH_RADIUS_KM,
) -> pd.DataFrame:
    """
    Combine WPI (+ optional GEM, OSM) into one terminal reference table.
    """
    wpi = wpi_df.copy().reset_index(drop=True)
    wpi["norm_name"] = wpi["port_name"].apply(normalise_name) if "port_name" in wpi.columns else ""

    terminals = []

    # --- WPI ports become "default" terminals (one per port) ---------------
    for i, row in wpi.iterrows():
        terminals.append({
            "terminal_id": f"WPI-{row.get('wpi_number', i)}",
            "terminal_name": row.get("port_name", f"WPI port {i}"),
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "wpi_port_name": row.get("port_name"),
            "wpi_number": row.get("wpi_number"),
            "harbor_type": row.get("harbor_type"),
            "harbor_use": row.get("harbor_use"),
            "gem_terminal_type": None,
            "gem_status": None,
            "osm_tags": None,
            "sources": "WPI",
        })

    # --- GEM terminals: attach to nearest WPI port within radius ----------
    if gem_df is not None and len(gem_df) and len(wpi):
        matched = nearest_neighbours(gem_df, wpi, k=1)
        for i, row in matched.iterrows():
            within = row["nearest_dist_km_0"] <= match_radius_km
            wpi_row = wpi.iloc[int(row["nearest_idx_0"])] if within else None
            terminals.append({
                "terminal_id": f"GEM-{i}",
                "terminal_name": row.get("terminal_name", f"GEM terminal {i}"),
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "wpi_port_name": wpi_row["port_name"] if within else None,
                "wpi_number": wpi_row.get("wpi_number") if within else None,
                "harbor_type": wpi_row.get("harbor_type") if within else None,
                "harbor_use": wpi_row.get("harbor_use") if within else None,
                "gem_terminal_type": row.get("terminal_type"),
                "gem_status": row.get("status"),
                "osm_tags": None,
                "sources": "GEM" + (" + WPI-matched" if within else " (no WPI match)"),
            })

    # --- OSM features: attach to nearest WPI port within radius -----------
    if osm_df is not None and len(osm_df) and len(wpi):
        matched = nearest_neighbours(osm_df, wpi, k=1)
        for i, row in matched.iterrows():
            within = row["nearest_dist_km_0"] <= match_radius_km
            wpi_row = wpi.iloc[int(row["nearest_idx_0"])] if within else None
            terminals.append({
                "terminal_id": f"OSM-{row.get('osm_id', i)}",
                "terminal_name": row.get("terminal_name") or f"OSM feature {row.get('osm_id', i)}",
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "wpi_port_name": wpi_row["port_name"] if within else None,
                "wpi_number": wpi_row.get("wpi_number") if within else None,
                "harbor_type": wpi_row.get("harbor_type") if within else None,
                "harbor_use": wpi_row.get("harbor_use") if within else None,
                "gem_terminal_type": None,
                "gem_status": None,
                "osm_tags": row.get("tags"),
                "sources": "OSM" + (" + WPI-matched" if within else " (no WPI match, possible offshore SBM)"),
            })

    out = pd.DataFrame(terminals)
    out["norm_name"] = out["terminal_name"].apply(normalise_name)

    seeds = out.apply(classify_terminal_seed, axis=1, result_type="expand")
    seeds.columns = ["seed_label", "seed_source"]
    out = pd.concat([out, seeds], axis=1)

    return out


def save_master_table(df: pd.DataFrame, filename: str = "master_terminals.parquet") -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / filename
    df.to_parquet(out_path, index=False)
    return out_path
