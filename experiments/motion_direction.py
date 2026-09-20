"""
Direction selectivity test: sweep the same moving bar across L1/L2/L3's real
hex-lattice positions in BOTH directions (increasing hex, then decreasing
hex), and compare how the real T4a-d (ON motion) / T5a-d (OFF motion)
subtypes respond in each direction.

T4/T5 themselves have no hex coordinates in this dataset (same situation as
R1-R6), so as before the bar is defined on L1/L2/L3's real positions and the
motion signal has to propagate through the real synapses onto Mi1/Mi4/Mi9
(T4's inputs) and Tm1/Tm2/Tm9 (T5's inputs) to reach T4/T5 -- nothing about
T4/T5 direction tuning is hand-coded, it's whatever the real wiring +
Dale's-principle signs produce.

Also tracks TmY18/TmY9b (real downstream targets of T4/T5 already in the
circuit) and the three descending neurons, to see whether directional
information survives all the way to a motor read-out.
"""
import sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from lif import LIFNetwork
import numpy as np

WEIGHT_SCALE = 0.013  # retuned for two-channel synaptic filtering (see fix notes)
DRIVE = 0.09
BAR_WIDTH = 5.0
DURATION_MS = 360

track_types = ['L1','L2','L3','T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d',
               'TmY18','TmY9b','DNg13','DNp11','DNpe002']

def run_sweep(direction):
    net = LIFNetwork.from_cache('motion_v1', weight_scale=WEIGHT_SCALE)
    neurons = net.neurons
    is_left = neurons['instance'].str.endswith('_L', na=False)
    lamina_mask = neurons['type'].isin(['L1','L2','L3']) & is_left & neurons['assignedOlHex1'].notna()
    lamina_idx = neurons.index[lamina_mask].to_numpy()
    lamina_hex = neurons.loc[lamina_mask, 'assignedOlHex1'].to_numpy()

    hex_min, hex_max = 1.0, 36.0
    dt = net.dt
    n_steps = int(round(DURATION_MS / dt))

    type_idx = {t: neurons.index[neurons['type'] == t].to_numpy() for t in track_types}
    counts_per_step = {t: np.zeros(n_steps, dtype=int) for t in track_types}
    bar_center_per_step = np.zeros(n_steps)

    net._synaptic_input = np.zeros(net.n)
    first_spike = {t: None for t in track_types}

    for t in range(n_steps):
        frac = t / n_steps
        if direction == 'forward':
            bar_center = hex_min + frac * (hex_max - hex_min)
        else:
            bar_center = hex_max - frac * (hex_max - hex_min)
        bar_center_per_step[t] = bar_center

        net.external_input[:] = 0.0
        under_bar = np.abs(lamina_hex - bar_center) <= (BAR_WIDTH / 2)
        net.external_input[lamina_idx[under_bar]] = DRIVE

        spikes = net.step()

        for ty in track_types:
            c = int(spikes[type_idx[ty]].sum())
            counts_per_step[ty][t] = c
            if c > 0 and first_spike[ty] is None:
                first_spike[ty] = t

    totals = {ty: int(counts_per_step[ty].sum()) for ty in track_types}
    return {
        "direction": direction,
        "bar_center_per_step": bar_center_per_step.tolist(),
        "counts_per_step": {t: counts_per_step[t].tolist() for t in track_types},
        "totals": totals,
        "first_spike": first_spike,
    }

print("Running forward sweep (increasing hex)...")
fwd = run_sweep('forward')
print("Totals:", fwd['totals'])
print()
print("Running backward sweep (decreasing hex)...")
bwd = run_sweep('backward')
print("Totals:", bwd['totals'])

print("\n--- Direction selectivity (forward total - backward total) per type ---")
for ty in track_types:
    f, b = fwd['totals'][ty], bwd['totals'][ty]
    diff = f - b
    denom = max(f + b, 1)
    di = diff / denom  # direction index, -1..1
    print(f"  {ty:8s} fwd={f:5d}  bwd={b:5d}  DSI={di:+.2f}")

out = {
    "duration_ms": DURATION_MS,
    "dt_ms": 1.0,
    "hex_min": 1.0, "hex_max": 36.0, "bar_width": BAR_WIDTH,
    "weight_scale": WEIGHT_SCALE,
    "forward": fwd,
    "backward": bwd,
}
with open('data/cache/motion_direction_result.json', 'w') as f:
    json.dump(out, f)
print("\nSaved data/cache/motion_direction_result.json")
