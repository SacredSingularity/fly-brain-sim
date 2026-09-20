"""
Direction selectivity, take 2: more statistical power.

The LIF network is fully deterministic -- rerunning the exact same sweep
gives the exact same spikes, so repeating a trial adds no information.
Instead this pools multiple *genuinely different* conditions (several sweep
speeds x several drive strengths, forward and backward at each), giving
both more total spikes to test on AND a check that any directional signal
is a real property of the circuit rather than a coincidence of one
particular speed/contrast setting.

Same circuit (motion_v1), same weight_scale as the retuned single-trial run
(0.013 -- the value that keeps T5 from saturating and T4 from staying
silent). Only DURATION_MS (sweep speed) and DRIVE (contrast) vary across
trials.
"""
import sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from lif import LIFNetwork
import numpy as np

WEIGHT_SCALE = 0.013
BAR_WIDTH = 5.0

# (duration_ms, drive) trial conditions: 3 speeds x 2 contrasts = 6 trials,
# each run forward and backward = 12 sweeps total.
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
    lamina_mask = neurons['type'].isin(['L1', 'L2', 'L3']) & is_left & neurons['assignedOlHex1'].notna()
    lamina_idx = neurons.index[lamina_mask].to_numpy()
    lamina_hex = neurons.loc[lamina_mask, 'assignedOlHex1'].to_numpy()

    hex_min, hex_max = 1.0, 36.0
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

print("\n--- Pooled totals across 6 conditions (12 sweeps) ---")
print("\n--- Direction selectivity index and binomial z per type ---")
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

# Also report per-trial DSI per type, to see whether sign is consistent
# across speeds/contrasts (a real effect should be) or flips around
# (more likely noise).
print("\n--- Per-trial T4 DSI (sign consistency check) ---")
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
    "weight_scale": WEIGHT_SCALE,
    "conditions": CONDITIONS,
    "per_trial": per_trial,
    "pooled_forward": pooled_fwd,
    "pooled_backward": pooled_bwd,
    "results": results,
}
with open('data/cache/motion_direction_multi_result.json', 'w') as f:
    json.dump(out, f)
print("\nSaved data/cache/motion_direction_multi_result.json")
