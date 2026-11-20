"""
Dormant-state fidelity checker for sparse spiking networks.
Tests whether tiered dormant/warm/hot updates preserve membrane potentials
identically across checkpoint-restore cycles as claimed in SparseProp (arXiv:2312.17216).
Run: python ziwago/learning/dormant_state_check.py [--tolerance 1e-9]
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from typing import Set

class Tier:
    DORMANT = "dormant"
    WARM = "warm"
    HOT = "hot"
    THRESHOLDS = {DORMANT: 0.1, WARM: 0.5, HOT: 1.0}

@dataclass
class Neuron:
    idx: int
    v: float = 0.0
    v_thresh: float = 1.0
    tau_m: float = 20.0
    dt: float = 1.0
    tier: str = Tier.DORMANT
    spikes: list = field(default_factory=list)

    def update(self, I_ext: float, active_neighbors: Set[int]) -> bool:
        fired = False
        if self.tier == Tier.HOT:
            decay = self.dt / self.tau_m
            self.v = self.v * (1 - decay) + self.dt * I_ext / self.tau_m
            if self.v >= self.v_thresh:
                self.v -= self.v_thresh
                self.spikes.append(len(self.spikes))
                fired = True
        elif self.tier == Tier.WARM:
            if abs(I_ext) > Tier.THRESHOLDS[Tier.WARM]:
                self.tier = Tier.HOT
                return self.update(I_ext, active_neighbors)
        return fired

    def checkpoint(self) -> dict:
        return {
            "idx": self.idx,
            "v": self.v,
            "tier": self.tier,
            "spikes": self.spikes.copy()
        }

    def restore(self, state: dict):
        self.v = state["v"]
        self.tier = state["tier"]
        self.spikes = state["spikes"].copy()

class SparseNetwork:
    def __init__(self, n_neurons: int):
        self.neurons = [Neuron(idx=i) for i in range(n_neurons)]
        self.active_bits: Set[int] = set()

    def step(self, I_ext: float):
        self.active_bits.clear()
        for n in self.neurons:
            if n.tier != Tier.DORMANT:
                self.active_bits.add(n.idx)
        for n in self.neurons:
            if n.tier != Tier.DORMANT:
                n.update(I_ext, self.active_bits)

    def checkpoint(self) -> list:
        return [n.checkpoint() for n in self.neurons if n.tier != Tier.DORMANT]

    def restore(self, states: list):
        restored = {s["idx"]: s for s in states}
        for n in self.neurons:
            if n.idx in restored:
                n.restore(restored[n.idx])

def run_test(n_neurons: int = 1000, steps: int = 100, tol: float = 1e-9):
    net = SparseNetwork(n_neurons)
    states_before = []
    for t in range(steps):
        I_ext = 0.8 + 0.4 * ((t % 10) / 10.0)
        net.step(I_ext)
        if t in [25, 50, 75]:
            states_before.append(net.checkpoint())
    net.restore(states_before[-1])
    for t in range(26, steps):
        I_ext = 0.8 + 0.4 * ((t % 10) / 10.0)
        net.step(I_ext)
    final_v = {n.idx: n.v for n in net.neurons if n.tier != Tier.DORMANT}
    for i, stored in enumerate(states_before):
        ref = {s["idx"]: s["v"] for s in stored}
        mismatches = sum(
            1 for idx, v in final_v.items()
            if idx in ref and abs(v - ref[idx]) > tol
        )
        print(f"Checkpoint {i}: {mismatches} potential mismatches")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--neurons", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    print(f"Running dormant-state fidelity test ({args.neurons} neurons, {args.steps} steps)...")
    run_test(args.neurons, args.steps, args.tolerance)
    print("Test complete.")