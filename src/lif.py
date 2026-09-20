"""
A minimal leaky integrate-and-fire (LIF) simulator for a connectome subgraph
built by graph.py.

This is deliberately simple: every neuron is treated as a point LIF unit.
Synapse weights from the connectome (proofread synapse counts) are used
directly as coupling strengths, scaled by a single constant. This is a
common simplifying choice for exploratory connectome simulations -- it
captures wiring topology and relative synapse strength, not the real
biophysics of every cell type (that would need per-neuron parameters
that aren't in this dataset).

Usage:
    from lif import LIFNetwork
    net = LIFNetwork.from_cache("visual_motor")
    net.set_input(neuron_type="R1-R6", rate_hz=50)
    spikes = net.run(duration_ms=200)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"


class LIFNetwork:
    def __init__(self, adj: sp.csr_matrix, neurons: pd.DataFrame, id_to_idx: dict[int, int],
                 dt_ms: float = 1.0, tau_ms: float = 20.0, v_thresh: float = 1.0,
                 v_reset: float = 0.0, refractory_ms: float = 2.0,
                 weight_scale: float = 0.002, baseline_current: float = 0.0,
                 tau_syn_exc_ms: float = 3.0, tau_syn_inh_ms: float = 12.0):
        """
        adj: sparse (n, n) matrix, adj[i, j] = synapse weight from neuron i -> neuron j
        neurons: dataframe with at least 'bodyId' and 'type' columns, row order matches adj.
            If it also has a 'sign' column (+1 excitatory / -1 inhibitory, from
            graph.py's neurotransmitter lookup), each neuron's outgoing synapses
            are signed by its own predicted transmitter (acetylcholine = +1;
            GABA, glutamate, histamine = -1). Without a 'sign' column, every
            neuron defaults to excitatory (+1) -- i.e. the old behavior.
        id_to_idx: bodyId -> row/col index
        dt_ms: simulation timestep
        tau_ms: membrane time constant (leak rate)
        v_thresh: spike threshold (arbitrary units, normalized)
        v_reset: reset potential after spike
        refractory_ms: refractory period after a spike
        weight_scale: scales raw synapse counts into membrane-potential units.
            This is the single most important tunable parameter -- there's no
            "correct" value from the connectome alone, since it doesn't encode
            per-synapse strength in physiological units. Start small and increase
            until you see plausible activity (not silence, not runaway firing).
        baseline_current: a small constant depolarizing drive applied to every
            neuron, representing background synaptic input from the rest of
            the brain that isn't included in this subgraph. Needed once real
            neurotransmitter signs are used: R1-R6 is inhibitory (histamine),
            so with baseline_current=0 an all-inhibitory external drive can
            only ever suppress activity, never produce any -- there's nothing
            for the inhibition to disinhibit. A small baseline gives neurons
            something to rest near threshold, so that a real fly-visual-system
            effect becomes visible: photoreceptor activity (~"dark") silences
            downstream neurons, and reduced photoreceptor activity (~"light")
            releases them from inhibition and lets the signal flow.
        tau_syn_exc_ms / tau_syn_inh_ms: separate synaptic decay time constants
            for excitatory vs inhibitory presynaptic input, replacing the old
            single-step "input is wiped and rebuilt every dt" propagation with
            a real exponentially-decaying synaptic current that persists across
            steps. This is a genuine biophysical asymmetry, not something tuned
            per-circuit: ionotropic ACh (nicotinic) channels are fast, while the
            GABA-A / glutamate-gated / histamine-gated chloride channels used by
            every inhibitory transmitter in this dataset are slower to close.
            Giving inhibition a longer decay than excitation is exactly the
            fast-excitation / slow-inhibition asymmetry a Hassenstein-Reichardt
            style motion correlator (e.g. T4/T5) needs to compare "now" against
            "recently" -- without it, excitatory and inhibitory input arrive and
            vanish in lockstep and can never encode direction, whatever the
            wiring says.
        """
        self.adj = adj
        self.n = adj.shape[0]
        self.neurons = neurons.reset_index(drop=True)
        self.id_to_idx = id_to_idx
        self.type_to_indices = self.neurons.groupby("type").apply(
            lambda g: g.index.to_numpy()
        ).to_dict()

        if "sign" in self.neurons.columns:
            self.sign = self.neurons["sign"].to_numpy(dtype=float)
        else:
            self.sign = np.ones(self.n)

        self.dt = dt_ms
        self.tau = tau_ms
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.refractory_steps = int(round(refractory_ms / dt_ms))
        self.weight_scale = weight_scale
        self.baseline_current = baseline_current

        # Split the adjacency by presynaptic sign so excitatory and inhibitory
        # input can be filtered through separate (fast / slow) synaptic
        # kinetics. adj is (pre, post); we slice rows (presynaptic side).
        self.exc_mask = self.sign > 0
        self.inh_mask = self.sign < 0
        self.adj_exc = self.adj[self.exc_mask, :].tocsr()
        self.adj_inh = self.adj[self.inh_mask, :].tocsr()
        self.tau_syn_exc = tau_syn_exc_ms
        self.tau_syn_inh = tau_syn_inh_ms
        self.decay_exc = np.exp(-self.dt / self.tau_syn_exc)
        self.decay_inh = np.exp(-self.dt / self.tau_syn_inh)

        self.reset_state()

    @classmethod
    def from_cache(cls, name: str, **kwargs) -> "LIFNetwork":
        adj = sp.load_npz(CACHE_DIR / f"{name}_adj.npz")
        neurons = pd.read_parquet(CACHE_DIR / f"{name}_neurons.parquet")
        with open(CACHE_DIR / f"{name}_index.json") as f:
            id_to_idx = {int(k): v for k, v in json.load(f).items()}
        return cls(adj, neurons, id_to_idx, **kwargs)

    def reset_state(self):
        self.v = np.zeros(self.n)
        self.refractory_countdown = np.zeros(self.n, dtype=int)
        self.external_input = np.zeros(self.n)
        # Persistent per-neuron synaptic currents, filtered separately for
        # excitatory (fast) and inhibitory (slow) input. _synaptic_input is
        # kept as their sum for backward compatibility with code that reads
        # it directly (e.g. experiment scripts print/inspect it).
        self.fast_channel = np.zeros(self.n)
        self.slow_channel = np.zeros(self.n)
        self._synaptic_input = self.fast_channel + self.slow_channel

    def set_input(self, neuron_type: str, rate_hz: float):
        """
        Drive all neurons of a given type with a constant input current,
        approximating a firing-rate-based external drive (e.g. a sensory
        stimulus hitting photoreceptors). rate_hz is converted to a rough
        current scale -- higher rate = stronger constant push toward threshold.
        """
        indices = self.type_to_indices.get(neuron_type)
        if indices is None or len(indices) == 0:
            raise ValueError(f"No neurons of type '{neuron_type}' in this network")
        # crude rate -> current mapping; tune alongside weight_scale
        current = rate_hz / 100.0
        self.external_input[indices] = current

    def clear_input(self):
        self.external_input[:] = 0.0

    def step(self) -> np.ndarray:
        """Advance the simulation by one dt. Returns boolean spike array for this step."""
        active = self.refractory_countdown <= 0

        # leak toward 0
        dv = (-self.v / self.tau) * self.dt
        self.v += dv * active

        # synaptic input: sum of incoming weighted spikes from last step
        # (adj is pre -> post, so incoming to neuron j = column j)
        # combined with constant external drive and background baseline
        self.v += (self._synaptic_input + self.external_input + self.baseline_current) * self.dt * active

        # count down refractory neurons
        self.refractory_countdown = np.maximum(self.refractory_countdown - 1, 0)

        # spikes
        spikes = (self.v >= self.v_thresh) & active
        self.v[spikes] = self.v_reset
        self.refractory_countdown[spikes] = self.refractory_steps

        # propagate this step's spikes as input for the NEXT step, split by
        # each spiking neuron's own predicted neurotransmitter (Dale's
        # principle: a neuron's sign is the same for all of its outgoing
        # synapses), and filtered through separate exponentially-decaying
        # synaptic currents -- fast for excitatory (ACh), slow for
        # inhibitory (GABA/glutamate/histamine chloride channels). Each
        # channel persists across steps (decays rather than being wiped),
        # which is what lets "recent past" and "right now" differ -- the
        # basic requirement for a Hassenstein-Reichardt-style correlator.
        exc_spikes = spikes[self.exc_mask].astype(float)
        inh_spikes = spikes[self.inh_mask].astype(float)
        self.fast_channel = (self.fast_channel * self.decay_exc
                              + self.adj_exc.T.dot(exc_spikes) * self.weight_scale)
        self.slow_channel = (self.slow_channel * self.decay_inh
                              - self.adj_inh.T.dot(inh_spikes) * self.weight_scale)
        self._synaptic_input = self.fast_channel + self.slow_channel

        return spikes

    def run(self, duration_ms: float) -> np.ndarray:
        """
        Run the simulation for duration_ms. Returns a (n_steps, n_neurons)
        boolean array of spikes.
        """
        self.fast_channel = np.zeros(self.n)
        self.slow_channel = np.zeros(self.n)
        self._synaptic_input = np.zeros(self.n)
        n_steps = int(round(duration_ms / self.dt))
        spike_log = np.zeros((n_steps, self.n), dtype=bool)
        for t in range(n_steps):
            spike_log[t] = self.step()
        return spike_log

    def spike_counts_by_type(self, spike_log: np.ndarray) -> pd.Series:
        counts = spike_log.sum(axis=0)
        s = pd.Series(counts, index=self.neurons["bodyId"])
        types = self.neurons.set_index("bodyId")["type"]
        return s.groupby(types).sum().sort_values(ascending=False)
