"""
Build a small circuit subgraph from the raw MaleCNS v1.0 connectome files.

Loads:
    data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather
        (211,577 neurons: bodyId, type, instance, side, etc.)
    data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather
        (151.8 million rows: body_pre, body_post, weight)

The weights file is ~1GB / 150M rows -- too big to load into memory at once
on a normal laptop. This script streams through it in batches and keeps
only the rows connecting neurons we actually care about.

Usage:
    python src/graph.py --types R1-R6 LoVP92 DNg13 --out visual_motor

Output (in data/cache/):
    <out>_neurons.parquet   - neuron metadata for the matched neurons
    <out>_edges.parquet     - filtered edge list (body_pre, body_post, weight)
    <out>_adj.npz           - scipy sparse weight matrix (local-indexed)
    <out>_index.json        - bodyId -> local index mapping
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "cache"

ANNOTATIONS_FILE = RAW_DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
WEIGHTS_FILE = RAW_DIR / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
NT_FILE = RAW_DIR / "body-neurotransmitters-male-cns-v1.0.feather"

# Sign each predicted neurotransmitter contributes at its release site.
# In the fly CNS: acetylcholine is excitatory (nicotinic ACh receptors);
# GABA and glutamate are predominantly inhibitory (GABA-A / glutamate-gated
# chloride channels, GluClalpha); histamine -- the photoreceptor
# transmitter -- is inhibitory via histamine-gated chloride channels
# (HisCl1/Ort). Dopamine/octopamine/serotonin are neuromodulatory and don't
# fit a clean +/-1 sign; they're rare in most circuits and treated as
# excitatory here as a conservative default. "unclear" (no confident
# prediction) also defaults to excitatory rather than silently dropping
# the connection.
NT_SIGN = {
    "acetylcholine": 1,
    "glutamate": -1,
    "gaba": -1,
    "histamine": -1,
    "dopamine": 1,
    "octopamine": 1,
    "serotonin": 1,
    "unclear": 1,
}


def load_neurotransmitters() -> pd.DataFrame:
    """
    Load per-neuron neurotransmitter predictions. Uses the cell-type-level
    consensus (celltype_predicted_nt) rather than the per-body consensus_nt,
    since individual bodies are often "unclear" while the type-level call
    (aggregated across all cells of that type) is usually confident.
    """
    if not NT_FILE.exists():
        return None
    nt = pd.read_feather(NT_FILE, columns=["body", "celltype_predicted_nt"])
    nt = nt.rename(columns={"body": "bodyId", "celltype_predicted_nt": "nt"})
    nt["sign"] = nt["nt"].map(NT_SIGN).fillna(1).astype(int)
    return nt


def load_neurons(types: list[str]) -> pd.DataFrame:
    """Load neuron annotations and filter to the requested type names."""
    ann = pd.read_feather(ANNOTATIONS_FILE)
    matched = ann[ann["type"].isin(types)].copy()
    if matched.empty:
        available_sample = ann["type"].dropna().unique()
        raise RuntimeError(
            f"No neurons found for types {types}. "
            f"Check spelling against Clio or the Cell Type Explorer. "
            f"(dataset has {len(available_sample)} distinct type names)"
        )
    print(f"Matched {len(matched)} neurons across {matched['type'].nunique()} types:")
    print(matched["type"].value_counts().to_string())

    nt = load_neurotransmitters()
    if nt is not None:
        matched = matched.merge(nt, on="bodyId", how="left")
        matched["nt"] = matched["nt"].fillna("unclear")
        matched["sign"] = matched["sign"].fillna(1).astype(int)
        print("\nNeurotransmitter / sign by type:")
        print(matched.groupby("type")[["nt", "sign"]]
              .agg(lambda x: x.mode().iat[0]).to_string())
    else:
        print("\nNo neurotransmitter file found -- all synapses treated as excitatory.")
        matched["nt"] = "unknown"
        matched["sign"] = 1

    return matched


def stream_filtered_edges(body_ids: set[int], batch_size_report: int = 200) -> pd.DataFrame:
    """
    Stream through the weights file in Arrow record batches, keeping only
    rows where BOTH body_pre and body_post are in body_ids.
    This avoids loading the full 150M-row table into memory.
    """
    kept_frames = []
    with pa.memory_map(str(WEIGHTS_FILE), "r") as source:
        reader = ipc.open_file(source)
        n_batches = reader.num_record_batches
        print(f"Scanning {n_batches} batches of the connectome weights file...")
        for i in range(n_batches):
            batch = reader.get_batch(i)
            df = batch.to_pandas()
            mask = df["body_pre"].isin(body_ids) & df["body_post"].isin(body_ids)
            if mask.any():
                kept_frames.append(df[mask])
            if (i + 1) % batch_size_report == 0 or i == n_batches - 1:
                print(f"  batch {i + 1}/{n_batches}, kept rows so far: "
                      f"{sum(len(f) for f in kept_frames)}")

    if not kept_frames:
        raise RuntimeError(
            "No edges found among the matched neurons. "
            "They may not be directly connected -- try a broader set of types."
        )
    edges = pd.concat(kept_frames, ignore_index=True)
    print(f"Total edges kept: {len(edges)}")
    return edges


def build_sparse_adjacency(neurons: pd.DataFrame, edges: pd.DataFrame):
    """Build a local-index sparse weight matrix from the filtered edge list."""
    body_ids = neurons["bodyId"].tolist()
    id_to_idx = {bid: i for i, bid in enumerate(body_ids)}

    rows = edges["body_pre"].map(id_to_idx)
    cols = edges["body_post"].map(id_to_idx)
    weights = edges["weight"].astype(float)

    n = len(body_ids)
    adj = sp.csr_matrix((weights, (rows, cols)), shape=(n, n))
    return adj, id_to_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--types", nargs="+", required=True,
                         help="Neuron type names to fetch, e.g. R1-R6 LoVP92 DNg13")
    parser.add_argument("--out", required=True, help="Output name prefix for cache files")
    args = parser.parse_args()

    if not ANNOTATIONS_FILE.exists() or not WEIGHTS_FILE.exists():
        raise RuntimeError(
            f"Expected raw data files not found in {RAW_DIR}. "
            "Make sure body-annotations-*.feather and connectome-weights-*.feather "
            "are placed in data/raw/."
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    neurons = load_neurons(args.types)
    body_ids = set(neurons["bodyId"].tolist())

    edges = stream_filtered_edges(body_ids)
    adj, id_to_idx = build_sparse_adjacency(neurons, edges)

    neurons.to_parquet(CACHE_DIR / f"{args.out}_neurons.parquet")
    edges.to_parquet(CACHE_DIR / f"{args.out}_edges.parquet")
    sp.save_npz(CACHE_DIR / f"{args.out}_adj.npz", adj)
    with open(CACHE_DIR / f"{args.out}_index.json", "w") as f:
        json.dump({str(k): v for k, v in id_to_idx.items()}, f)

    print(f"\nSaved to {CACHE_DIR}/{args.out}_*")
    print(f"  neurons: {len(neurons)}")
    print(f"  edges: {adj.nnz}")
    print(f"  density: {adj.nnz / (adj.shape[0] ** 2):.4f}")


if __name__ == "__main__":
    main()
