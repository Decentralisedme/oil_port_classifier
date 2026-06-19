"""
load_data.py
-------------
Readers/normalisers for the four input sources:

  1. World Port Index (WPI)        -> data/raw/wpi*.csv
  2. Global Energy Monitor (GEM)   -> data/raw/gem_oil_infrastructure*.xlsx
  3. OpenStreetMap (Overpass API)  -> live query, cached to data/raw/osm_oil_terminals.json
  4. AIS feed (your own DB export) -> data/raw/ais_positions*.jsonl / .parquet
                                       data/raw/ais_static*.jsonl / .parquet

Design notes:
  - WPI and GEM are downloaded MANUALLY by you (both require navigating
    a download form / free account). This module just reads whatever
    you place in data/raw/.
  - OSM is queried live via the public Overpass API (read-only GET,
    no auth, no scraping of disallowed sites).
  - AIS readers expect either newline-delimited JSON (one message per
    line, matching the PositionReport / ShipStaticData shapes you
    showed) or a parquet file with equivalent columns. Parquet is
    strongly recommended for anything beyond a quick test - it's far
    faster and smaller for AIS volumes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


# ---------------------------------------------------------------------------
# World Port Index
# ---------------------------------------------------------------------------

# Common column-name variants seen across WPI export versions.
_WPI_COLUMN_MAP = {
    "port_name": "port_name",
    "main_port_name": "port_name",
    "portname": "port_name",
    "world_port_index_number": "wpi_number",
    "index_number": "wpi_number",
    "wpi_number": "wpi_number",
    "country_code": "country",
    "country": "country",
    "latitude": "latitude",
    "lat": "latitude",
    "longitude": "longitude",
    "lon": "longitude",
    "long": "longitude",
    "harbor_size": "harbor_size",
    "harborsize": "harbor_size",
    "harbor_type": "harbor_type",
    "harbortype": "harbor_type",
    "harbor_use": "harbor_use",
    "harboruse": "harbor_use",
}


def load_wpi(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load a World Port Index CSV export.

    Download from NGA MSI: https://msi.nga.mil/Publications/WPI
    (CSV / shapefile export - use the CSV).

    Returns a dataframe with at least:
      port_name, wpi_number, country, latitude, longitude,
      harbor_size, harbor_type, harbor_use
    (plus any other original columns, lowercased).
    """
    if path is None:
        candidates = sorted(RAW_DIR.glob("*wpi*.csv")) + sorted(RAW_DIR.glob("*WPI*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No WPI csv found in {RAW_DIR}. "
                "Download from https://msi.nga.mil/Publications/WPI and "
                "place the CSV there (filename containing 'wpi')."
            )
        path = candidates[0]

    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rename = {c: _WPI_COLUMN_MAP.get(c, c) for c in df.columns}
    df = df.rename(columns=rename)

    required = {"latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"WPI file missing expected columns {missing}. Found: {list(df.columns)}")

    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    df["source"] = "WPI"
    return df


# ---------------------------------------------------------------------------
# Global Energy Monitor - Global Oil Infrastructure Tracker
# ---------------------------------------------------------------------------

_GEM_COLUMN_MAP = {
    "unit_name": "terminal_name",
    "terminal_name": "terminal_name",
    "wiki_name": "terminal_name",
    "country": "country",
    "latitude": "latitude",
    "longitude": "longitude",
    "status": "status",
    "terminal_type": "terminal_type",
    "type": "terminal_type",
    "capacity_(bbl)": "capacity_bbl",
    "capacity_bbl": "capacity_bbl",
    "owner": "owner",
}


def load_gem(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load a Global Energy Monitor Oil & Gas terminal/infrastructure export.

    Download (free account required):
      https://globalenergymonitor.org/projects/global-oil-infrastructure-tracker/

    Accepts .xlsx or .csv. Returns dataframe with at least:
      terminal_name, country, latitude, longitude, terminal_type, status
    """
    if path is None:
        candidates = sorted(RAW_DIR.glob("*gem*.xlsx")) + sorted(RAW_DIR.glob("*GEM*.xlsx"))
        candidates += sorted(RAW_DIR.glob("*gem*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No GEM file found in {RAW_DIR}. Download the Global Oil "
                "Infrastructure Tracker from globalenergymonitor.org and "
                "place it there (filename containing 'gem')."
            )
        path = candidates[0]

    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rename = {c: _GEM_COLUMN_MAP.get(c, c) for c in df.columns}
    df = df.rename(columns=rename)

    required = {"latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"GEM file missing expected columns {missing}. Found: {list(df.columns)}")

    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    df["source"] = "GEM"
    return df


# ---------------------------------------------------------------------------
# OpenStreetMap (Overpass API)
# ---------------------------------------------------------------------------

# Heuristic tag set for oil terminals / SBMs / offshore moorings.
# These are the tags most commonly used in practice; OSM coverage of
# offshore infrastructure is patchy, so treat this as a *supplementary*
# signal, not a primary source.
_OVERPASS_QUERY_TEMPLATE = """
[out:json][timeout:180];
(
  node["man_made"="petroleum_well"]{bbox};
  node["man_made"="pipeline"]["substance"~"oil|crude|petroleum",i]{bbox};
  way["industrial"="oil"]{bbox};
  way["landuse"="industrial"]["product"~"oil|petroleum|crude",i]{bbox};
  node["seamark:type"="mooring"]{bbox};
  way["seamark:type"="mooring"]{bbox};
  node["man_made"="mooring"]{bbox};
  node["harbour"="yes"]["name"~"oil|petroleum|terminal|tanker",i]{bbox};
  way["man_made"="storage_tank"]["content"~"oil|petroleum|crude",i]{bbox};
);
out center tags;
"""


def query_osm_oil_terminals(bbox: tuple[float, float, float, float] | None = None,
                             cache_name: str = "osm_oil_terminals.json",
                             force_refresh: bool = False) -> pd.DataFrame:
    """
    Query the public Overpass API for oil-terminal-related infrastructure.

    bbox: (south, west, north, east) in degrees. If None, queries the
          whole world - this is SLOW and may be rejected by the public
          Overpass instance for large areas. Prefer running this per
          region of interest.

    Results are cached to data/raw/<cache_name> so repeated runs don't
    hammer the public API.
    """
    cache_path = RAW_DIR / cache_name
    if cache_path.exists() and not force_refresh:
        with open(cache_path) as f:
            data = json.load(f)
    else:
        bbox_str = f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})" if bbox else ""
        query = _OVERPASS_QUERY_TEMPLATE.format(bbox=bbox_str)
        resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=200)
        resp.raise_for_status()
        data = resp.json()
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(data, f)

    rows = []
    for el in data.get("elements", []):
        if "center" in el:
            lat, lon = el["center"]["lat"], el["center"]["lon"]
        elif "lat" in el:
            lat, lon = el["lat"], el["lon"]
        else:
            continue
        tags = el.get("tags", {})
        rows.append({
            "osm_id": el.get("id"),
            "osm_type": el.get("type"),
            "terminal_name": tags.get("name", ""),
            "latitude": lat,
            "longitude": lon,
            "tags": json.dumps(tags),
            "source": "OSM",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# AIS data (your own feed)
# ---------------------------------------------------------------------------

POSITION_COLUMNS = [
    "UserID", "Latitude", "Longitude", "Sog", "Cog", "TrueHeading",
    "NavigationalStatus", "RateOfTurn", "Timestamp", "MsgTimestamp",
]

STATIC_COLUMNS = [
    "UserID", "ImoNumber", "CallSign", "Name", "Type",
    "MaximumStaticDraught", "Destination", "Eta", "MsgTimestamp",
]


def _read_jsonl_or_parquet(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_ais_positions(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load AIS PositionReport messages.

    Expects each record to contain (at least): UserID, Latitude,
    Longitude, Sog, NavigationalStatus, and SOME timestamp field
    (Timestamp is the AIS-internal seconds field, 0-59 - it is NOT a
    usable absolute time. If your DB export adds an absolute capture
    timestamp, e.g. 'MsgTimestamp' / 'received_at', keep it - the
    classifier needs real elapsed time to detect "stationary for N
    hours").
    """
    if path is None:
        candidates = sorted(RAW_DIR.glob("*position*"))
        if not candidates:
            raise FileNotFoundError(f"No AIS position file found in {RAW_DIR}")
        path = candidates[0]
    path = Path(path)
    df = _read_jsonl_or_parquet(path)
    df.columns = [c for c in df.columns]  # preserve original casing (AIS convention)
    return df


def load_ais_static(path: str | Path | None = None) -> pd.DataFrame:
    """Load AIS ShipStaticData messages (vessel type, IMO, draught, name, etc.)."""
    if path is None:
        candidates = sorted(RAW_DIR.glob("*static*"))
        if not candidates:
            raise FileNotFoundError(f"No AIS static file found in {RAW_DIR}")
        path = candidates[0]
    path = Path(path)
    df = _read_jsonl_or_parquet(path)
    return df
