"""
Fetch a targeted circuit subgraph from the MaleCNS v1.0 connectome via neuPrint,
and cache it locally as a sparse adjacency matrix + neuron index.

Usage:
    export NEUPRINT_TOKEN="your_token_here"
    python fetch.py --types R1 R2 R3 R4 R5 R6 LoVP92 DNg13 --out visual_motor

This pulls the named neuron types, all synaptic connections between them,
and saves:
    data/cache/<out>_neurons.parquet   - neuron metadata (id, type, instance, side, etc.)
    data/cache/<out>_edges.parquet     - edge list (bodyId_pre, bodyId_post, weight, roi)
    data/cache/<out>_adj.npz           - scipy sparse weight matrix (indexed by local id)
    data/cache/<out>_index.json        - bodyId <-> local index mapping
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from neuprint import Client, fetch_neurons, fetch_adjacencies, NeuronCriteria as NC

DATASET = "male-cns:v1.0"
SERVER = "https://neuprint.janelia.org"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def get_client(token: str | None = None) -> Client:
    token = token or os.environ.get("NEUPRINT_TOKEN")
    if not token:
        raise RuntimeError(
            "No neuPrint token found. Set NEUPRINT_TOKEN env var or pass --token.\n"
            "Get one by logging into https://neuprint.janelia.org and visiting your Account page."
        )
    return Client(SERVER, dataset=DATASET, token=token)


def fetch_circuit(types: list[str], client: Client):
    """Fetch neurons matching the given type names and all edges among them."""
    print(f"Fetching neuron metadata for types: {types}")
    neurons, roi_info = fetch_neurons(NC(type=types))
    if neurons.empty:
        raise RuntimeError(
            f"No neurons found for types {types}. "
            "Check spelling/case against the Cell Type Explorer or Clio before retrying."
        )
    print(f"  found {len(neurons)} neurons")

    body_ids = neurons.bodyId.tolist()

    print("Fetching adjacencies among these neurons...")
    edges, _ = fetch_adjacencies(body_ids, body_ids, client=client)
    print(f"  found {len(edges)} edges")

    return neurons, edges


def build_sparse_adjacency(neurons: pd.DataFrame, edges: pd.DataFrame):
    """Build a local-index sparse weight matrix from a neuPrint edge list."""
    body_ids = neurons.bodyId.tolist()
    id_to_idx = {bid: i for i, bid in enumerate(body_ids)}

    # neuprint-python's fetch_adjacencies returns columns like:
    # bodyId_pre, bodyId_post, weight (summed across ROIs) when roi_info is aggregated.
    # If per-ROI rows are returned instead, group first.
    if "roi" in edges.columns:
        agg = edges.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()
    else:
        agg = edges

    rows = agg.bodyId_pre.map(id_to_idx)
    cols = agg.bodyId_post.map(id_to_idx)
    weights = agg.weight.astype(float)

    n = len(body_ids)
    adj = sp.csr_matrix((weights, (rows, cols)), shape=(n, n))
    return adj, id_to_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--types", nargs="+", required=True, help="Neuron type names to fetch")
    parser.add_argument("--out", required=True, help="Output name prefix for cache files")
    parser.add_argument("--token", default=None, help="neuPrint API token (or set NEUPRINT_TOKEN)")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    client = get_client(args.token)

    neurons, edges = fetch_circuit(args.types, client)
    adj, id_to_idx = build_sparse_adjacency(neurons, edges)

    neurons.to_parquet(CACHE_DIR / f"{args.out}_neurons.parquet")
    edges.to_parquet(CACHE_DIR / f"{args.out}_edges.parquet")
    sp.save_npz(CACHE_DIR / f"{args.out}_adj.npz", adj)
    with open(CACHE_DIR / f"{args.out}_index.json", "w") as f:
        json.dump({str(k): v for k, v in id_to_idx.items()}, f)

    print(f"\nSaved to {CACHE_DIR}/{args.out}_*")
    print(f"  neurons: {len(neurons)}")
    print(f"  edges (aggregated): {adj.nnz}")
    print(f"  density: {adj.nnz / (adj.shape[0] ** 2):.4f}")


if __name__ == "__main__":
    main()
