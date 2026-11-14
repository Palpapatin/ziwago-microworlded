#!/usr/bin/env python3
"""B.0: localize the ACCESS VIOLATION during full-organism construction.

Replicates ZiwagoOrganism.__init__ step by step with faulthandler enabled and a
background RSS monitor, so the native crash stack and the memory trajectory are
captured even when the process dies. No checkpoint is loaded (construction only).
Run ONLY via tools/limited_run.py.
"""
from __future__ import annotations

import faulthandler
import sys
import threading
import time
from pathlib import Path

ROOT = Path(r"C:\Users\yarma\Desktop\Ziwago18")
sys.path.insert(0, str(ROOT))

faulthandler.enable(all_threads=True)

import psutil  # noqa: E402

_PROC = psutil.Process()
_T0 = time.perf_counter()
_STOP = False
_PEAK = 0.0


def _rss_gb() -> float:
    return _PROC.memory_info().rss / 1e9


def _log(msg: str) -> None:
    print(f"[b0 +{time.perf_counter()-_T0:6.1f}s rss={_rss_gb():.3f}GB] {msg}", flush=True)


def _monitor() -> None:
    global _PEAK
    while not _STOP:
        r = _rss_gb()
        if r > _PEAK:
            _PEAK = r
        print(f"[mon +{time.perf_counter()-_T0:6.1f}s rss={r:.3f}GB peak={_PEAK:.3f}GB]", flush=True)
        time.sleep(0.1)


def main() -> int:
    import torch
    torch.set_num_threads(4)
    _log(f"start; free={psutil.virtual_memory().available/1e9:.2f}GB torch={torch.__version__}")
    mon = threading.Thread(target=_monitor, daemon=True)
    mon.start()

    _log("import v14 core class")
    from train_ziwago_v14_pure_synaptic_dialogue import ZiwagoPureSynapticDialogueOrganismV14
    _log("import genesis modules")
    from ziwago.cortex.genesis_engine import (
        BA10CounterfactualInterventionSimulator, BidirectionalMeaningClosureVerifier,
        CorticalRelationalWorldGraph, EmergentOntologicalRoleDiscovery,
        MDThalamusSynapticSpeechActRouter, SyntacticValenceWaveGenerator,
        ThetaGammaSpikingPhaseMultiplexer,
    )

    _log("STEP 1: construct v14 core (cortex MoE)")
    core = ZiwagoPureSynapticDialogueOrganismV14(root=ROOT, workers=2, cpu_threads=4, seed=42)
    core.use_ngrams = False
    _log("STEP 1 done: v14 core constructed")

    _log("STEP 2: SyntacticValenceWaveGenerator(d_dend=8192) -> genesis_engine.py:51 torch.zeros(8192,2048)")
    valence = SyntacticValenceWaveGenerator(d_vec=64, d_dend=8192, top_k_wta=32, seed=42 + 1801)
    _log("STEP 2 done: valence wave constructed")

    _log("STEP 3: CorticalRelationalWorldGraph")
    world = CorticalRelationalWorldGraph(
        encode_fn=core.encode_text, compute_vec64_fn=core.compute_stimulus_vec64,
        get_vec64_fn=core.get_vec64, decode_fn=core.decode_token)
    _log("STEP 3 done: world graph")

    _log("STEP 4: EmergentOntologicalRoleDiscovery(n_bins=2048)")
    onto = EmergentOntologicalRoleDiscovery(n_bins=2048)
    _log("STEP 4 done: ontology")

    _log("STEP 5: ThetaGammaSpikingPhaseMultiplexer")
    tg = ThetaGammaSpikingPhaseMultiplexer(d_vec=64, n_gamma_slots=6, bins_per_gamma=20)
    _log("STEP 5 done: theta-gamma")

    _log("STEP 6: closure verifier + BA10 counterfactual")
    closure = BidirectionalMeaningClosureVerifier()
    cf = BA10CounterfactualInterventionSimulator()
    _log("STEP 6 done")

    _log("STEP 7: MDThalamusSynapticSpeechActRouter")
    router = MDThalamusSynapticSpeechActRouter(
        encode_fn=core.encode_text,
        compute_sdr_fn=core.dendritic_plateau_readout._compute_cortical_sdr2048)
    _log("STEP 7 done: thalamic router")

    _log("ALL CONSTRUCTION STEPS PASSED")
    global _STOP
    _STOP = True
    mon.join(timeout=2)
    _log(f"final rss={_rss_gb():.3f}GB")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        _log(f"FAILED: {type(exc).__name__}: {exc}; sampled peak={_PEAK:.3f}GB")
        raise