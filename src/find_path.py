"""
Discover which neuron types actually bridge two sets of neuron types in the
connectome, by scanning the weights file for direct synaptic partners.

Why this exists: a circuit like "R1-R6 -> LoVP92 -> DNg13" named in a paper
or press release is usually a multi-hop pathway, not a direct wire. If you
build a graph.py subgraph using only the named types, you'll often get zero
edges between the start and end groups -- the real signal passes through
unnamed intermediate cell types first. This script finds candidates for
those intermediates so you can add them to your circuit.

Usage:
    python src/find_path.py --from R1-R6 --to LoVP92 DNg13 --hops 2

This finds:
    - types that receive input directly from --from types (1 hop downstream)
    - types that send input directly to --to types (1 hop upstream)
    - the overlap between those two sets = likely bridging types
"""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
ANNOTATIONS_FILE = RAW_DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
WEIGHTS_FILE = RAW_DIR / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"


def load_type_map() -> pd.DataFrame:
    ann = pd.read_feather(ANNOTATIONS_FILE, columns=["bodyId", "type"])
    return ann


def stream_partners(seed_ids: set[int], direction: str, min_weight: int = 3) -> pd.DataFrame:
    """
    direction='downstream': find rows where body_pre is in seed_ids -> return body_post partners
    direction='upstream':   find rows where body_post is in seed_ids -> return body_pre partners
    Filters out very weak connections (min_weight) to cut noise.
    """
    kept = []
    with pa.memory_map(str(WEIGHTS_FILE), "r") as source:
        reader = ipc.open_file(source)
        n_batches = reader.num_record_batches
        for i in range(n_batches):
            df = reader.get_batch(i).to_pandas()
            df = df[df["weight"] >= min_weight]
            if direction == "downstream":
                mask = df["body_pre"].isin(seed_ids)
                partners = df.loc[mask, ["body_post", "weight"]].rename(columns={"body_post": "bodyId"})
            else:
                mask = df["body_post"].isin(seed_ids)
                partners = df.loc[mask, ["body_pre", "weight"]].rename(columns={"body_pre": "bodyId"})
            if not partners.empty:
                kept.append(partners)
            if (i + 1) % 400 == 0 or i == n_batches - 1:
                print(f"  scanned {i + 1}/{n_batches} batches...")
    if not kept:
        return pd.DataFrame(columns=["bodyId", "weight"])
    return pd.concat(kept, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_types", nargs="+", required=True)
    parser.add_argument("--to", dest="to_types", nargs="+", required=True)
    parser.add_argument("--min-weight", type=int, default=3,
                         help="Ignore synapses weaker than this (default 3)")
    parser.add_argument("--top", type=int, default=25,
                         help="Show top N candidate bridging types by total synapse weight")
    args = parser.parse_args()

    ann = load_type_map()

    from_ids = set(ann.loc[ann["type"].isin(args.from_types), "bodyId"])
    to_ids = set(ann.loc[ann["type"].isin(args.to_types), "bodyId"])
    print(f"'from' types {args.from_types}: {len(from_ids)} neurons")
    print(f"'to' types {args.to_types}: {len(to_ids)} neurons")

    print("\nScanning for neurons downstream of 'from' types...")
    downstream = stream_partners(from_ids, "downstream", args.min_weight)
    downstream = downstream.merge(ann, on="bodyId", how="left")
    downstream_types = (downstream.groupby("type")["weight"].sum()
                         .sort_values(ascending=False))

    print("\nScanning for neurons upstream of 'to' types...")
    upstream = stream_partners(to_ids, "upstream", args.min_weight)
    upstream = upstream.merge(ann, on="bodyId", how="left")
    upstream_types = (upstream.groupby("type")["weight"].sum()
                       .sort_values(ascending=False))

    bridging_types = set(downstream_types.index) & set(upstream_types.index)
    bridging_types -= set(args.from_types) | set(args.to_types)

    print(f"\n=== Top {args.top} types directly downstream of {args.from_types} ===")
    print(downstream_types.head(args.top).to_string())

    print(f"\n=== Top {args.top} types directly upstream of {args.to_types} ===")
    print(upstream_types.head(args.top).to_string())

    print(f"\n=== Candidate bridging types (appear in both lists, {len(bridging_types)} found) ===")
    if bridging_types:
        combined = (downstream_types.reindex(bridging_types).fillna(0) +
                    upstream_types.reindex(bridging_types).fillna(0)).sort_values(ascending=False)
        print(combined.head(args.top).to_string())
        print(f"\nSuggested graph.py command:")
        suggested_types = list(args.from_types) + list(combined.head(10).index) + list(args.to_types)
        print(f"python src/graph.py --types {' '.join(suggested_types)} --out visual_motor_v2")
    else:
        print("None found directly -- the path may be more than 2 hops. "
              "Try increasing --min-weight down, or run this again using one "
              "of the downstream/upstream types as a new seed to go one hop further.")


if __name__ == "__main__":
    main()
