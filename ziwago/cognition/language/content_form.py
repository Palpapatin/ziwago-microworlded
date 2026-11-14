"""Content/form separation with causal theta-gamma temporal binding (stage F).

Blueprint §3.10 (CSGL) / §7 F. The content plan (role -> grounded filler with
provenance) is separated from form realization. Role fillers are temporally bound
into a theta-gamma TTFS frame; the DECODED vectors -- not the raw inputs -- drive
content-addressable filler recovery and constrained surface rendering. Temporal
binding is causal: event loss below an availability gate, or a filler that cannot
be recovered, refuses composition rather than fabricating content. Closure verifies
every recovered role survives in the rendered output. Compose (a novel role
combination) is distinguished from replay (an exact stored episode).

Opt-in: enabled=False is neutral (compose returns None). This is an engineering
model, not biophysical fidelity or a neural proof of truth; no n-grams, no global
backprop. The valence form-realization engine (SyntacticValenceWaveGenerator) is the
existing render_fn hook and is not duplicated here.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set

import torch
import torch.nn.functional as F

from ziwago.cortex.genesis_engine import ThetaGammaSpikingPhaseMultiplexer


@dataclass
class RoleFiller:
    """One grounded content slot: a role, its intended filler label, the filler
    vector bound into the theta frame, and the evidence source (provenance)."""
    role: str
    label: str
    vector: torch.Tensor
    source: str = "user"


class ContentFormComposer:
    ROLE_SLOT_NAMES = ThetaGammaSpikingPhaseMultiplexer.ROLE_SLOT_NAMES

    def __init__(self, d_vec: int = 64, n_gamma_slots: int = 6, bins_per_gamma: int = 20,
                 value_scale: float = 0.5, availability_threshold: float = 0.5,
                 recovery_threshold: float = 0.5, closure_threshold: float = 1.0,
                 enabled: bool = False, allow_dynamic_roles: bool = False,
                 multiplexer: Optional[ThetaGammaSpikingPhaseMultiplexer] = None):
        if int(d_vec) < 1 or not math.isfinite(value_scale) or value_scale <= 0:
            raise ValueError("d_vec must be positive and value_scale finite positive.")
        if not (0.0 <= availability_threshold <= 1.0 and 0.0 <= recovery_threshold <= 1.0
                and 0.0 <= closure_threshold <= 1.0):
            raise ValueError("Thresholds must lie in [0, 1].")
        self.d_vec = int(d_vec)
        self.value_scale = float(value_scale)
        self.availability_threshold = float(availability_threshold)
        self.recovery_threshold = float(recovery_threshold)
        self.closure_threshold = float(closure_threshold)
        self.enabled = bool(enabled)
        self.multiplexer = multiplexer if multiplexer is not None else ThetaGammaSpikingPhaseMultiplexer(
            d_vec=self.d_vec, n_gamma_slots=int(n_gamma_slots),
            bins_per_gamma=int(bins_per_gamma), value_scale=self.value_scale)
        if self.multiplexer.d_vec != self.d_vec:
            raise ValueError("Multiplexer d_vec must match the composer.")
        self.last_report: Dict[str, object] = {}

    @staticmethod
    def build_role_dictionary_from_bindings(
        role_bindings: Dict[str, str],
        encode_fn: Callable[[str], torch.Tensor | Sequence[int] | List[int]],
        d_vec: int = 64,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Convert a learned role map (`filler -> role`) into a role->label->vector dictionary.

        The map is canonical-slot oriented in the binder (`novel_filler -> canonical_role`),
        but the composer expects the reverse direction: `role -> {label: vector}`.
        """
        vocab: Dict[str, Dict[str, torch.Tensor]] = {}
        for label, role in role_bindings.items():
            label_norm = str(label).strip()
            role_norm = str(role).strip()
            if not label_norm or not role_norm:
                continue
            encoded = encode_fn(label_norm)
            if isinstance(encoded, torch.Tensor):
                vec = encoded.detach().float().reshape(-1)
            elif isinstance(encoded, (list, tuple)):
                vec = torch.as_tensor(list(encoded), dtype=torch.float32).reshape(-1)
            else:
                continue
            if vec.numel() == 0:
                continue
            if vec.numel() < int(d_vec):
                vec = F.pad(vec, (0, int(d_vec) - vec.numel()))
            vec = vec[:int(d_vec)]
            vocab.setdefault(role_norm, {})[label_norm] = vec
        return vocab

    def _recover(self, decoded: torch.Tensor, vocab_role: Dict[str, torch.Tensor]
                 ) -> tuple[Optional[str], float]:
        """Content-addressable recovery: nearest vocabulary vector by cosine.

        Silence decodes to a zero vector (cosine 0) and recovers nothing, so a lost
        filler cannot be silently replaced by an arbitrary vocabulary entry.
        """
        q = F.normalize(decoded, p=2, dim=0)
        best_label: Optional[str] = None
        best_cos = -1.0
        for label, vec in vocab_role.items():
            v = F.normalize(vec.to(device=q.device, dtype=q.dtype), p=2, dim=0)
            cos = float(torch.dot(q, v))
            if cos > best_cos:
                best_cos, best_label = cos, label
        if best_label is None or best_cos < self.recovery_threshold:
            return None, best_cos
        return best_label, best_cos

    def _refusal(self, reason: str, **metrics: object) -> Dict[str, object]:
        report = {"composed": False, "reason": reason, **metrics}
        self.last_report = report
        return report

    def compose(self, fillers: Sequence[RoleFiller],
                vocabulary: Dict[str, Dict[str, torch.Tensor]],
                render_fn: Callable[[Dict[str, str]], str],
                known_episodes: Optional[Set[frozenset]] = None) -> Optional[Dict[str, object]]:
        """Compose a surface form from a grounded content plan via theta-gamma binding.

        Returns None when disabled (opt-in neutral). Otherwise a report dict with
        composed=True (text, content, provenance, fidelity, closure, mode) or
        composed=False with an explicit refusal reason. No filler is fabricated.
        """
        if not self.enabled:
            return None
        plan: List[RoleFiller] = list(fillers)
        if not plan:
            return self._refusal("empty_content")
        names = self.ROLE_SLOT_NAMES[:self.multiplexer.n_gamma_slots]
        seen: set[str] = set()
        role_vectors: Dict[str, torch.Tensor] = {}
        for f in plan:
            if f.role not in names:
                raise ValueError(f"Unknown or unconfigured role slot: {f.role}")
            if f.role in seen:
                raise ValueError(f"Duplicate role slot: {f.role}")
            seen.add(f.role)
            if not f.label or not str(f.source).strip():
                raise ValueError("Filler label and provenance source must be non-empty.")
            vec = f.vector.detach().to(device="cpu", dtype=torch.float32)
            if vec.shape != (self.d_vec,) or not bool(torch.isfinite(vec).all()):
                raise ValueError("Filler vector must be finite and match d_vec.")
            role_vectors[f.role] = vec

        raster, windows = self.multiplexer.encode_roles_into_theta_spike_train(role_vectors)
        decoded, report = self.multiplexer.decode_roles_from_theta_spike_train(raster, return_report=True)
        # Availability over the SUPPLIED roles only: intentional absent slots are not
        # transmission loss (the multiplexer cannot distinguish them without a mask).
        avail = [float(report["roles"][f.role]["availability_fraction"]) for f in plan]
        availability = sum(avail) / len(avail)
        if availability < self.availability_threshold:
            return self._refusal("temporal_loss", availability_fraction=availability,
                                 loss_fraction=1.0 - availability)

        content: Dict[str, str] = {}
        recovery_cosine: Dict[str, float] = {}
        fidelity_hits = 0
        for f in plan:
            label, cos = self._recover(decoded[f.role], vocabulary.get(f.role, {}))
            recovery_cosine[f.role] = round(cos, 4)
            if label is None:
                return self._refusal("filler_lost", role=f.role,
                                     availability_fraction=availability, recovery_cosine=recovery_cosine)
            content[f.role] = label
            if label == f.label:
                fidelity_hits += 1
        fidelity = fidelity_hits / len(plan)

        text = render_fn(dict(content))
        covered = sum(1 for label in content.values()
                      if re.search(r"(?<!\w)" + re.escape(label) + r"(?!\w)", text, flags=re.I))
        closure = covered / len(content)
        if closure < self.closure_threshold:
            return self._refusal("closure_failed", closure_score=closure,
                                 recovery_fidelity=fidelity, content=dict(content))

        signature = frozenset(content.items())
        mode = "replay" if (known_episodes is not None and signature in known_episodes) else "compose"
        result: Dict[str, object] = {
            "composed": True, "text": text, "content": dict(content),
            "composition_mode": mode, "availability_fraction": availability,
            "loss_fraction": 1.0 - availability, "recovery_fidelity": fidelity,
            "closure_score": closure, "recovery_cosine": recovery_cosine,
            "provenance": {f.role: str(f.source) for f in plan}, "theta_windows": windows,
        }
        self.last_report = result
        return result

    def telemetry(self) -> Dict[str, object]:
        return {
            "enabled": self.enabled, "d_vec": self.d_vec,
            "availability_threshold": self.availability_threshold,
            "recovery_threshold": self.recovery_threshold,
            "closure_threshold": self.closure_threshold,
            "multiplexer": self.multiplexer.export_state_dict(),
            "last_composed": self.last_report.get("composed"),
            "last_mode": self.last_report.get("composition_mode"),
            "last_reason": self.last_report.get("reason"),
        }

    def export_state_dict(self) -> Dict[str, object]:
        return {
            "d_vec": self.d_vec, "value_scale": self.value_scale,
            "availability_threshold": self.availability_threshold,
            "recovery_threshold": self.recovery_threshold,
            "closure_threshold": self.closure_threshold, "enabled": self.enabled,
            "multiplexer": self.multiplexer.export_state_dict(),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        restored = type(self)(
            d_vec=int(state["d_vec"]), value_scale=float(state["value_scale"]),
            availability_threshold=float(state["availability_threshold"]),
            recovery_threshold=float(state["recovery_threshold"]),
            closure_threshold=float(state["closure_threshold"]),
            enabled=bool(state["enabled"]))
        restored.multiplexer.load_state_dict(state["multiplexer"])
        self.__dict__.update(restored.__dict__)