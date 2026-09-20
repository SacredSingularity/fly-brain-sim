"""
First experiment: drive the R1-R6 photoreceptors with a constant "light on"
input and watch activity propagate through the LoVP92 interneurons to the
DNg13 descending motor neuron.

Run from the project root:
    python experiments/run_visual_motor.py

Requires data/cache/visual_motor_* files, built first with:
    python src/graph.py --types R1-R6 LoVP92 DNg13 --out visual_motor
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from lif import LIFNetwork  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "cache"


def main():
    net = LIFNetwork.from_cache("visual_motor", weight_scale=0.002)

    print(f"Network loaded: {net.n} neurons")
    print(net.neurons["type"].value_counts().to_string())

    # Simulate a "light on" stimulus hitting the photoreceptors
    net.set_input(neuron_type="R1-R6", rate_hz=80)

    duration_ms = 300
    spikes = net.run(duration_ms=duration_ms)

    print(f"\nSpike counts by neuron type over {duration_ms}ms:")
    print(net.spike_counts_by_type(spikes).to_string())

    # Export for the visualizer: one row per (time_ms, bodyId) spike event,
    # plus neuron metadata, as JSON the HTML page can load directly.
    spike_times, neuron_idx = spikes.nonzero()
    body_ids = net.neurons["bodyId"].to_numpy()
    types = net.neurons["type"].to_numpy()

    events = [
        {"t": int(t) * net.dt, "bodyId": int(body_ids[i]), "type": str(types[i])}
        for t, i in zip(spike_times, neuron_idx)
    ]

    neuron_list = [
        {"bodyId": int(row.bodyId), "type": str(row.type),
         "instance": str(row.instance) if pd.notna(row.instance) else None}
        for row in net.neurons.itertuples()
    ]

    export = {
        "duration_ms": duration_ms,
        "dt_ms": net.dt,
        "neurons": neuron_list,
        "events": events,
    }

    out_path = OUT_DIR / "visual_motor_spikes.json"
    with open(out_path, "w") as f:
        json.dump(export, f)
    print(f"\nExported {len(events)} spike events to {out_path}")


if __name__ == "__main__":
    import pandas as pd  # noqa: E402
    main()
