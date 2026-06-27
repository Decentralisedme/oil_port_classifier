"""
run_pipeline.py
----------------
End-to-end orchestration. Run from the project root:

    python src/run_pipeline.py

Expects:
  data/raw/<something with 'wpi' in the name>.csv|.shp|.zip   (required)
  data/raw/osm_oil_terminals.json         (optional, auto-fetched if osm_bbox given)
  data/raw/<something with 'position' in the name>.parquet|.jsonl  (required for classification)
  data/raw/<something with 'static' in the name>.parquet|.jsonl    (required for classification)

NOTE - GEM/GOIT dropped:
  GEM's "Global Oil Infrastructure Tracker" download contains only
  pipeline-segment rows (PipelineName, SegmentName, StartLocation,
  EndLocation, LengthKm...) with NO latitude/longitude columns.
  It cannot be used as a terminal point source. Terminal seeding is
  handled entirely by data/raw/manual_seed_terminals.csv (25 ports)
  and WPI for the geofencing backbone.

Produces:
  data/processed/master_terminals.parquet
  outputs/stops_classified.csv
  outputs/port_classification.csv
  outputs/sts_candidates.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from load_data import load_wpi, query_osm_oil_terminals, load_ais_positions, load_ais_static
from clean_ports import build_master_terminal_table, save_master_table, apply_manual_overrides, apply_manual_seed_terminals
from classify import (
    detect_stationary_periods,
    assign_terminal,
    attach_terminal_seed,
    apply_voyage_sequence_inference,
    apply_terminal_majority_pass,
    aggregate_port_classification,
    detect_sts_events,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def main(
    osm_bbox: tuple[float, float, float, float] | None = None,
    ais_time_col: str = "MsgTimestamp",
):
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    # 1. Reference data ------------------------------------------------------
    print("Loading WPI...")
    wpi = load_wpi()
    print(f"  {len(wpi)} ports loaded")

    # GEM/GOIT not used - contains pipeline segments only, no terminal coordinates.
    # Terminal seed labels come from data/raw/manual_seed_terminals.csv instead.

    osm = None
    if osm_bbox is not None:
        print("Querying OSM (Overpass)...")
        osm = query_osm_oil_terminals(bbox=osm_bbox)
        print(f"  {len(osm)} features")
    else:
        print("Skipping OSM (no bbox supplied to main()).")

    print("Building master terminal table...")
    terminals = build_master_terminal_table(wpi, gem_df=None, osm_df=osm)
    print("Applying manual terminal-label overrides (name-matched)...")
    terminals = apply_manual_overrides(terminals)
    print("Applying manual seed terminals (coordinate-matched)...")
    terminals = apply_manual_seed_terminals(terminals)
    save_master_table(terminals)
    print(f"  {len(terminals)} terminals -> data/processed/master_terminals.parquet")
    n_seeded = (terminals["seed_label"] != "UNKNOWN").sum()
    print(f"  {n_seeded}/{len(terminals)} terminals have a seed_label (LOAD/DISCHARGE/BOTH/STS_HUB)")

    # 2. AIS-derived classification ------------------------------------------
    try:
        positions = load_ais_positions()
        static = load_ais_static()
    except FileNotFoundError as e:
        print(f"AIS data not found, stopping before classification step: {e}")
        return

    print(f"Loaded {len(positions)} position reports, {len(static)} static reports.")

    print("Detecting stationary periods...")
    stops = detect_stationary_periods(positions, time_col=ais_time_col)
    print(f"  {len(stops)} stops found")

    print("Assigning terminals (geofence)...")
    stops = assign_terminal(stops, terminals)

    print("Attaching terminal seed labels...")
    stops = attach_terminal_seed(stops, terminals)

    print("Applying voyage-sequence alternation inference...")
    stops = apply_voyage_sequence_inference(stops)

    print("Applying cross-vessel terminal-majority pass...")
    stops = apply_terminal_majority_pass(stops)

    stops.to_csv(OUTPUTS / "stops_classified.csv", index=False)
    print(f"  -> outputs/stops_classified.csv ({len(stops)} port calls)")

    print("Aggregating to port-level classification...")
    port_classification = aggregate_port_classification(stops)
    port_classification.to_csv(OUTPUTS / "port_classification.csv", index=False)
    print(f"  -> outputs/port_classification.csv ({len(port_classification)} terminals)")

    print("Detecting STS candidates...")
    sts = detect_sts_events(positions, terminals, static_data=static, time_col=ais_time_col)
    sts.to_csv(OUTPUTS / "sts_candidates.csv", index=False)
    print(f"  -> outputs/sts_candidates.csv ({len(sts)} candidate events)")

    print("\nDone.")


if __name__ == "__main__":
    main()