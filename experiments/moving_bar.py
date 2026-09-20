"""
Dime-varying visual input: sweep a real moving bar across the retinotopic
hex-lattice coordinates of L1/L2/L3 (the lamina), instead of driving the
whole population with one flat constant. Uses each neuron's real
assignedOlHex1 position (from the annotations file) to decide, at every
1ms timestep, whether it currently falls "under" the bar.

R1-R6 lacks hex coordinates in this dataset (photoreceptors converge
multiple-to-one onto each lamina column), but L1/L2/L3 are retinotopically
mapped one column per ommatidium, so they're used directly as the
"sensory surface" here -- a real, data-derived spatial layout, not an
invented one.
"""
import sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from lif import LIFNetwork
import numpy as np

net = LIFNetwork.from_cache('visual_motor_v4', weight_scale=0.06)
neurons = net.neurons
is_left = neurons['instance'].str.endswith('_L', na=False)

lamina_mask = neurons['type'].isin(['L1', 'L2', 'L3']) & is_left & neurons['assignedOlHex1'].notna()
lamina_idx = neurons.index[lamina_mask].to_numpy()
lamina_hex = neurons.loc[lamina_mask, 'assignedOlHex1'].to_numpy()

hex_min, hex_max = 1.0, 36.0
bar_width = 5.0
drive_current = 0.09   # a bit above our earlier tonic 0.07, since only ~14% of the population is ever "lit" at once

duration_ms = 360
dt = net.dt
n_steps = int(round(duration_ms / dt))

# track per-type, per-timestep spike counts for later visualization
track_types = ['L1', 'L2', 'L3', 'Tm3', 'Tm4', 'Tm20', 'Tm5c', 'LoVP92', 'DNg13', 'DNp11', 'DNpe002']
type_idx = {t: neurons.index[neurons['type'] == t].to_numpy() for t in track_types}
counts_per_step = {t: np.zeros(n_steps, dtype=int) for t in track_types}
bar_center_per_step = np.zeros(n_steps)

net._synaptic_input = np.zeros(net.n)
first_spike = {t: None for t in track_types}

for t in range(n_steps):
    frac = t / n_steps
    bar_center = hex_min + frac * (hex_max - hex_min)
    bar_center_per_step[t] = bar_center

    net.external_input[:] = 0.0
    under_bar = np.abs(lamina_hex - bar_center) <= (bar_width / 2)
    net.external_input[lamina_idx[under_bar]] = drive_current

    spikes = net.step()

    for ty in track_types:
        c = int(spikes[type_idx[ty]].sum())
        counts_per_step[ty][t] = c
        if c > 0 and first_spike[ty] is None:
            first_spike[ty] = t

print("Total spikes by tracked type:")
for ty in track_types:
    print(f"  {ty:8s} total={counts_per_step[ty].sum():6d}  first_spike_t={first_spike[ty]}")

out = {
    "duration_ms": duration_ms,
    "dt_ms": dt,
    "bar_center_per_step": bar_center_per_step.tolist(),
    "hex_min": hex_min, "hex_max": hex_max, "bar_width": bar_width,
    "counts_per_step": {t: counts_per_step[t].tolist() for t in track_types},
    "first_spike": first_spike,
}
with open('data/cache/moving_bar_result.json', 'w') as f:
    json.dump(out, f)
print("\nSaved data/cache/moving_bar_result.json")
