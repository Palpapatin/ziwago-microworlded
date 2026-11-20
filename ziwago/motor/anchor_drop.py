"""Anchor-Drop Toolkit - Commission C#58

What carries after the anchor is dropped?
Two mechanisms: deterministic replay proves identity without trust,
and hash-comparison detects drift even when seeding is broken.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

VERSION = "0.1.0"
DEFAULT_RUNS = 5


class Checkpoint:
    def __init__(self, state: Dict[str, Any], seed: int):
        self.state = state
        self.seed = seed
        self.hash = self._hash()

    def _hash(self) -> str:
        raw = json.dumps({"s": self.seed, "st": self.state}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def carry(self) -> bool:
        return True

    def drift_detected(self, other: "Checkpoint") -> bool:
        return self.hash != other.hash


def run_sim(seed: int, n_steps: int = 100) -> Tuple[Checkpoint, List[float]]:
    rng = random.Random(seed)
    values = [rng.gauss(0, 1) for _ in range(n_steps)]
    acc = sum(values)
    cp = Checkpoint({"acc": acc, "n": n_steps, "seed": seed}, seed)
    return cp, values


def test_deterministic():
    seeds = [42, 17, 99]
    results = {s: [] for s in seeds}
    for s in seeds:
        cps = [run_sim(s)[0] for _ in range(DEFAULT_RUNS)]
        results[s] = [c.hash for c in cps]
    identical = all(len(set(hashes)) == 1 for hashes in results.values())
    return identical, results


def test_drift_detection():
    cp1, _ = run_sim(42)
    corrupted = Checkpoint({"acc": cp1.state["acc"] + 1e-6, "n": 100, "seed": 42}, 42)
    return cp1.drift_detected(corrupted)


def main():
    parser = argparse.ArgumentParser(description="Anchor-Drop Toolkit")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--mode", choices=["deterministic", "drift", "all"], default="all")
    args = parser.parse_args()

    out: Dict[str, Any] = {"version": VERSION, "commission": "C#58"}

    if args.mode in ("deterministic", "all"):
        identical, results = test_deterministic()
        out["deterministic_replay"] = {
            "identical_hashes": identical,
            "results": results
        }

    if args.mode in ("drift", "all"):
        detected = test_drift_detection()
        out["drift_detection"] = {"detected": detected}

    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
