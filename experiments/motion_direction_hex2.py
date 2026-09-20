"""
Refinement of the direction-selectivity finding: sweep the SECOND real hex
axis (assignedOlHex2), orthogonal to the one used in
motion_direction_multi.py (assignedOlHex1).

Motivation: on the hex1 axis, T4a/T4c/T4d came out reproducibly direction-
selective but T4b stayed near zero. Real fly T4a-d are each tuned to one of
four cardinal directions around the visual field, roughly 90 degrees apart
-- so a subtype that's silent on one axis is exactly what you'd expect if
it's actually tuned to the orthogonal axis instead. This sweeps hex2 with
the same pooling-across-conditions design used for hex1, to see whether
T4b (and to check the others aren't secretly also tuned here) lights up.

Nothing about tuning is hand-coded here either: same circuit (motion_v1),
same two-channel synaptic filtering, same weight_scale -- only which real
spatial coordinate drives the bar changes.
"""
import sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from lif import LIFNetwork
import numpy as np

WEIGHT_SCALE = 0.013
BAR_WIDTH = 5.0

CONDITIONS = [
    (240, 0.09), (360, 0.09), (540, 0.09),
    (240, 0.12), (360, 0.12), (540, 0.12),
]

track_types = ['L1', 'L2', 'L3', 'T4a', 'T4b', 'T4c', 'T4d',
               'T5a', 'T5b', 'T5c', 'T5d', 'TmY18', 'TmY9b',
               'DNg13', 'DNp11', 'DNpe002']


def run_sweep(direction, duration_ms, drive):
    net = LIFNetwork.from_cache('motion_v1', weight_scale=WEIGHT_SCALE)
    neurons = net.neurons
    is_left = neurons['instance'].str.endswith('_L', na=False)
    lamina_mask = (neurons['type'].isin(['L1', 'L2', 'L3']) & is_left
                   & neurons['assignedOlHex2'].notna())
    lamina_idx = neurons.index[lamina_mask].to_numpy()
    lamina_hex = neurons.loc[lamina_mask, 'assignedOlHex2'].to_numpy()

    hex_min, hex_max = 1.0, 39.0
    dt = net.dt
    n_steps = int(round(duration_ms / dt))

    type_idx = {t: neurons.index[neurons['type'] == t].to_numpy() for t in track_types}
    totals = {t: 0 for t in track_types}

    for t in range(n_steps):
        frac = t / n_steps
        if direction == 'forward':
            bar_center = hex_min + frac * (hex_max - hex_min)
        else:
            bar_center = hex_max - frac * (hex_max - hex_min)

        net.external_input[:] = 0.0
        under_bar = np.abs(lamina_hex - bar_center) <= (BAR_WIDTH / 2)
        net.external_input[lamina_idx[under_bar]] = drive

        spikes = net.step()
        for ty in track_types:
            totals[ty] += int(spikes[type_idx[ty]].sum())

    return totals


pooled_fwd = {t: 0 for t in track_types}
pooled_bwd = {t: 0 for t in track_types}
per_trial = []

for duration_ms, drive in CONDITIONS:
    print(f"Trial: duration={duration_ms}ms drive={drive} ...")
    f = run_sweep('forward', duration_ms, drive)
    b = run_sweep('backward', duration_ms, drive)
    per_trial.append({"duration_ms": duration_ms, "drive": drive, "forward": f, "backward": b})
    for ty in track_types:
        pooled_fwd[ty] += f[ty]
        pooled_bwd[ty] += b[ty]

print("\n--- hex2-axis direction selectivity (pooled across 6 conditions) ---")
import math
results = {}
for ty in track_types:
    f, b = pooled_fwd[ty], pooled_bwd[ty]
    n = f + b
    denom = max(n, 1)
    di = (f - b) / denom
    z = (f - n / 2) / math.sqrt(n * 0.25) if n > 0 else 0.0
    results[ty] = {"fwd": f, "bwd": b, "n": n, "dsi": di, "z": z}
    print(f"  {ty:8s} fwd={f:6d}  bwd={b:6d}  n={n:6d}  DSI={di:+.3f}  z={z:+.2f}")

print("\n--- Per-trial T4 DSI on hex2 (sign consistency check) ---")
for trial in per_trial:
    line = f"  dur={trial['duration_ms']:4d} drive={trial['drive']:.2f}: "
    parts = []
    for ty in ['T4a', 'T4b', 'T4c', 'T4d']:
        f, b = trial['forward'][ty], trial['backward'][ty]
        n = f + b
        di = (f - b) / n if n > 0 else 0.0
        parts.append(f"{ty}={di:+.2f}(n={n})")
    print(line + " ".join(parts))

out = {
    "axis": "assignedOlHex2",
    "weight_scale": WEIGHT_SCALE,
    "conditions": CONDITIONS,
    "per_trial": per_trial,
    "pooled_forward": pooled_fwd,
    "pooled_backward": pooled_bwd,
    "results": results,
}
with open('data/cache/motion_direction_hex2_result.json', 'w') as f:
    json.dump(out, f)
print("\nSaved data/cache/motion_direction_hex2_result.json")
