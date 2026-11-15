#!/usr/bin/env python3
"""Local knowledge extraction, experimental plasticity and measured trace diagnostics."""

from __future__ import annotations

import copy
import csv
import hashlib
import math
import re
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# ============================================================================
# ЭТАП 1: ЗОЛОТОЙ ФОНД ЗНАНИЙ (`Knowledge/` Pipeline & `KnowledgeDistillery`)
# ============================================================================
class KnowledgeDistillery:
    """Extract source-backed prose and well-formed tables; never synthesize fixtures."""

    def __init__(self, knowledge_dir: Optional[Path] = None):
        self.knowledge_dir = Path(knowledge_dir) if knowledge_dir is not None else (
            Path(__file__).resolve().parents[2] / "data" / "Knowledge"
        )
        self.parsed_files_summary: Dict[str, int] = {}
        self.records: List[dict] = []
        self.extraction_errors: List[str] = []
        self.extracted_chemical_agents_count = 0
        self.available = self.knowledge_dir.is_dir()
        self.status = "available" if self.available else "unavailable"
        self._scan_knowledge_directory()

    @property
    def extracted_records_count(self) -> int:
        return len(self.records)

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"[*`]", "", text).strip()

    def _scan_knowledge_directory(self) -> None:
        if not self.available:
            return
        chemical_names = set()
        for path in sorted(self.knowledge_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".csv", ".tsv"}:
                continue
            source = path.relative_to(self.knowledge_dir).as_posix()
            try:
                raw = path.read_bytes()
                text = raw.decode("utf-8-sig", errors="strict")
            except (OSError, UnicodeError) as exc:
                self.extraction_errors.append(f"{source}: {exc}")
                continue
            source_hash = hashlib.sha256(raw).hexdigest()
            self.parsed_files_summary[source] = len(text)
            section = path.stem

            def record(kind, line, content, **extra):
                content = " ".join(content.split())
                if not content:
                    return
                identity = f"{source}:{source_hash}:{line}:{kind}:{content}"
                self.records.append({
                    "record_id": hashlib.sha256(identity.encode()).hexdigest(),
                    "source": source, "source_hash": source_hash, "line": line,
                    "kind": kind, "section": section, "content": content, **extra,
                })

            def table_row(headers, cells, line):
                if len(cells) != len(headers) or any(not c for c in cells):
                    self.extraction_errors.append(f"{source}:{line}: malformed table row")
                    return
                fields = dict(zip(headers, cells))
                record("table", line, "; ".join(f"{h}: {v}" for h, v in fields.items()), fields=fields)
                agent_col = next((i for i, h in enumerate(headers) if re.search(
                    r"agent|chemical|compound|mediator|вещество|агент|медиатор", h, re.I
                )), None)
                if agent_col is None and re.search(r"chemical|хим", source, re.I):
                    agent_col = next((i for i, h in enumerate(headers) if h.lower() in {
                        "name", "название", "название вещества", "нейромедиатор"
                    }), None)
                    if agent_col is None and re.match(r"^\d+[.)]\s*\S", cells[0]):
                        agent_col = 0
                if agent_col is not None:
                    name = re.sub(r"^\d+[.)]?\s*", "", cells[agent_col]).strip()
                    if name and not name.isdigit():
                        chemical_names.add(name.casefold())

            if path.suffix.lower() in {".csv", ".tsv"}:
                try:
                    rows = list(csv.reader(text.splitlines(), delimiter="\t" if path.suffix.lower() == ".tsv" else ",", strict=True))
                    headers = [self._clean(c) for c in rows[0]] if rows else []
                    if len(headers) < 2 or any(not h for h in headers) or len(set(headers)) != len(headers):
                        self.extraction_errors.append(f"{source}: invalid table header")
                        continue
                    for line, cells in enumerate(rows[1:], 2):
                        if cells:
                            table_row(headers, [self._clean(c) for c in cells], line)
                except csv.Error as exc:
                    self.extraction_errors.append(f"{source}: {exc}")
                continue

            lines = text.splitlines()
            i, in_code = 0, False
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith(("```", "~~~")):
                    in_code = not in_code
                    i += 1
                    continue
                if in_code or not line:
                    i += 1
                    continue
                if line.startswith("#"):
                    section = line.lstrip("# ").strip()
                    i += 1
                    continue
                if "|" in line:
                    headers = [self._clean(c) for c in line.strip("|").split("|")]
                    sep = lines[i + 1].strip() if i + 1 < len(lines) else ""
                    separators = [c.strip() for c in sep.strip("|").split("|")]
                    valid = (len(headers) >= 2 and all(headers) and len(set(headers)) == len(headers)
                             and len(separators) == len(headers)
                             and all(re.fullmatch(r":?-{3,}:?", c) for c in separators))
                    if not valid:
                        self.extraction_errors.append(f"{source}:{i + 1}: invalid markdown table")
                        i += 1
                        continue
                    i += 2
                    while i < len(lines) and "|" in lines[i]:
                        cells = [self._clean(c) for c in lines[i].strip().strip("|").split("|")]
                        table_row(headers, cells, i + 1)
                        i += 1
                    continue
                start = i + 1
                paragraph = []
                while i < len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(("#", "```", "~~~")) and "|" not in lines[i]:
                    paragraph.append(re.sub(r"^\s*(?:[-+]\s+|\d+[.)]\s+)", "", lines[i]))
                    i += 1
                for sentence in re.split(r"(?<=[.!?])\s+", self._clean(" ".join(paragraph))):
                    record("text", start, sentence)
        self.extracted_chemical_agents_count = len(chemical_names)

    def get_knowledge_core_lessons(self) -> List[Tuple[str, str, str]]:
        return [(f"knowledge_{Path(r['source']).stem}", f"What does {r['section']} state?", r["content"])
                for r in self.records]

    def get_knowledge_world_graph_facts(self) -> List[str]:
        return [r["content"] for r in self.records]


# ============================================================================
# ПРОРЫВ 1 (ЭТАП 2): DYNAMIC NEUROMODULATORY HOMEOSTASIS & BCM PLASTICITY GATE
# ============================================================================
class DynamicNeuromodulatoryBCMGate:
    """Activity-dependent novelty gate with a persistent sliding BCM threshold."""

    def __init__(
        self,
        d_sdr: int = 2048,
        alpha_ne: float = 2.4,
        beta_ach: float = 2.2,
        gamma_red: float = 2.8,
        tau_bcm: float = 50.0,
    ):
        self.d_sdr = int(d_sdr)
        self.alpha_ne = float(alpha_ne)
        self.beta_ach = float(beta_ach)
        self.gamma_red = float(gamma_red)
        self.tau_bcm = float(tau_bcm)
        if self.d_sdr <= 0 or not all(math.isfinite(v) for v in (
            self.alpha_ne, self.beta_ach, self.gamma_red, self.tau_bcm
        )) or self.tau_bcm <= 0:
            raise ValueError("invalid BCM configuration")
        self.theta_bcm = torch.full((self.d_sdr,), 0.08, dtype=torch.float32)
        self.running_centroid_sdr = torch.zeros(self.d_sdr, dtype=torch.float32)
        self.total_evaluated: int = 0
        self.total_accepted: int = 0

    def export_state_dict(self) -> dict:
        return {"d_sdr": self.d_sdr, "alpha_ne": self.alpha_ne, "beta_ach": self.beta_ach,
                "gamma_red": self.gamma_red, "tau_bcm": self.tau_bcm,
                "theta_bcm": self.theta_bcm.detach().cpu().clone(),
                "running_centroid_sdr": self.running_centroid_sdr.detach().cpu().clone(),
                "total_evaluated": self.total_evaluated, "total_accepted": self.total_accepted}

    def load_state_dict(self, state: dict) -> None:
        d = int(state["d_sdr"])
        theta = torch.as_tensor(state["theta_bcm"]).detach().float().cpu().clone()
        centroid = torch.as_tensor(state["running_centroid_sdr"]).detach().float().cpu().clone()
        if d <= 0 or theta.shape != (d,) or centroid.shape != (d,) or not (
            torch.isfinite(theta).all() and torch.isfinite(centroid).all()
        ):
            raise ValueError("invalid BCM state")
        candidate = type(self)(d, **{k: float(state[k]) for k in ("alpha_ne", "beta_ach", "gamma_red", "tau_bcm")})
        evaluated, accepted = int(state["total_evaluated"]), int(state["total_accepted"])
        if not 0 <= accepted <= evaluated or (theta < 0).any():
            raise ValueError("invalid BCM counters or threshold")
        candidate.theta_bcm, candidate.running_centroid_sdr = theta, centroid
        candidate.total_evaluated, candidate.total_accepted = evaluated, accepted
        self.__dict__.update(candidate.__dict__)

    def evaluate_neuromodulatory_gate(
        self,
        sdr_vec: torch.Tensor,
        causal_density_hint: float = 1.0,
        grammar_ok: bool = True,
    ) -> Dict[str, object]:
        """Evaluate novelty/surprise and update the sliding activity threshold."""
        if sdr_vec.shape != (self.d_sdr,) or not torch.isfinite(sdr_vec).all():
            raise ValueError("expected a finite SDR vector")
        self.theta_bcm = self.theta_bcm.to(device=sdr_vec.device)
        self.running_centroid_sdr = self.running_centroid_sdr.to(device=sdr_vec.device)
        self.total_evaluated += 1
        if not grammar_ok:
            return {
                "accepted": False,
                "gate_prob": 0.0,
                "ach_novelty": 0.0,
                "ne_surprise": 0.0,
                "redundancy": 1.0,
                "bcm_gain": 0.0,
            }

        with torch.no_grad():
            y = F.normalize(sdr_vec.float().clamp(min=0.0) + 1e-8, p=2, dim=0)
            if float(self.running_centroid_sdr.norm().item()) < 1e-5:
                redundancy = 0.0
            else:
                c_norm = F.normalize(self.running_centroid_sdr + 1e-8, p=2, dim=0)
                redundancy = float(torch.dot(y, c_norm).clamp(min=0.0, max=1.0).item())

            ach_novelty = max(0.0, 1.0 - redundancy)
            # Эпистемический сюрприз NE усиливается при появлении редких терминов (превышающих скользящий порог BCM theta_M)
            bcm_delta = y * (y - self.theta_bcm)
            ne_surprise = float(bcm_delta.clamp(min=0.0).sum().item()) * 1.8 + 0.35 * float(causal_density_hint)

            logit = (
                self.alpha_ne * ne_surprise
                + self.beta_ach * ach_novelty
                - self.gamma_red * (redundancy ** 2)
                - 1.35
            )
            gate_prob = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit))))
            accepted = bool(gate_prob >= 0.52 and redundancy < 0.92)

            # Обновляем скользящий гомеостатический порог BCM: theta_M <- (1 - 1/tau)*theta_M + (1/tau)*y^2
            inv_tau = 1.0 / max(1.0, self.tau_bcm)
            self.theta_bcm.mul_(1.0 - inv_tau).add_(inv_tau * (y * y))
            self.running_centroid_sdr.mul_(0.98).add_(0.02 * y)

            if accepted:
                self.total_accepted += 1

            return {
                "accepted": accepted,
                "gate_prob": round(gate_prob, 4),
                "ach_novelty": round(ach_novelty, 4),
                "ne_surprise": round(ne_surprise, 4),
                "redundancy": round(redundancy, 4),
                "bcm_gain": round(float(bcm_delta.clamp(min=0.0).sum().item()), 4),
            }

    @property
    def ingestion_yield_pct(self) -> float:
        if self.total_evaluated == 0:
            return 0.0
        return round(100.0 * float(self.total_accepted) / float(self.total_evaluated), 3)


# ============================================================================
# ПРОРЫВ 2 (ЭТАП 3): ДВУХКОМПАРТМЕНТНЫЕ ПИРАМИДНЫЕ НЕЙРОНЫ L2/3–L5 & BAC-FIRING (Sacramento / Larkum)
# ============================================================================
class TwoCompartmentLarkumMicrocircuit:
    """Two-vector basal/apical coincidence model with a learned SST predictor."""

    def __init__(self, d_vec: int = 64, theta_ca: float = 0.35):
        self.d_vec = int(d_vec)
        self.theta_ca = float(theta_ca)
        if self.d_vec <= 0 or not math.isfinite(self.theta_ca) or self.theta_ca < 0:
            raise ValueError("invalid microcircuit configuration")
        # Матрица дендритных интернейронов соматостатина SST, предсказывающих top-down вход
        self.w_sst_pred = torch.eye(self.d_vec, dtype=torch.float32) * 0.25
        # Матрица базально-соматической проекции Urbanczik-Senn
        self.w_basal_soma = torch.eye(self.d_vec, dtype=torch.float32)

    def export_state_dict(self) -> dict:
        return {"d_vec": self.d_vec, "theta_ca": self.theta_ca,
                "w_sst_pred": self.w_sst_pred.detach().cpu().clone(),
                "w_basal_soma": self.w_basal_soma.detach().cpu().clone()}

    def load_state_dict(self, state: dict) -> None:
        d = int(state["d_vec"])
        weights = [torch.as_tensor(state[k]).detach().float().cpu().clone()
                   for k in ("w_sst_pred", "w_basal_soma")]
        if d <= 0 or any(w.shape != (d, d) or not torch.isfinite(w).all() for w in weights):
            raise ValueError("invalid microcircuit state")
        candidate = type(self)(d, float(state["theta_ca"]))
        candidate.w_sst_pred, candidate.w_basal_soma = weights
        self.__dict__.update(candidate.__dict__)

    def step_microcircuit(
        self,
        x_basal: torch.Tensor,
        x_apical_topdown: torch.Tensor,
        dt_coincidence_ms: float = 3.5,
        learn_sst: bool = True,
    ) -> Dict[str, object]:
        if x_basal.shape != (self.d_vec,) or x_apical_topdown.shape != (self.d_vec,) or not (
            torch.isfinite(x_basal).all() and torch.isfinite(x_apical_topdown).all()
        ):
            raise ValueError("expected finite basal and apical vectors")
        self.w_sst_pred = self.w_sst_pred.to(device=x_basal.device)
        self.w_basal_soma = self.w_basal_soma.to(device=x_basal.device)
        x_apical_topdown = x_apical_topdown.to(device=x_basal.device)
        with torch.no_grad():
            v_basal = F.normalize(x_basal.float() @ self.w_basal_soma + 1e-8, p=2, dim=-1)
            v_top = F.normalize(x_apical_topdown.float() + 1e-8, p=2, dim=-1)

            # Тормозное предсказание интернейронов соматостатина SST (Sacramento 2018):
            sst_inhibition = v_basal @ self.w_sst_pred
            apical_prediction_error = v_top - sst_inhibition

            # Амплитуда апикального кальциевого тока I_Ca (Larkum 1999) с условием Jahr-Stevens Mg2+ разблокировки:
            v_depol = float(torch.dot(v_basal, v_top).clamp(min=0.0).item())
            mg_unblock = 1.0 / (1.0 + (1.2 / 3.57) * math.exp(-3.5 * v_depol))
            apical_ca_drive = float(v_top.norm().item()) * mg_unblock * (0.5 + 0.5 * v_depol)

            bap_active = float(v_basal.norm().item()) > 0.5
            apical_plateau_active = apical_ca_drive > self.theta_ca
            bac_burst_flag = bool(bap_active and apical_plateau_active and dt_coincidence_ms <= 5.0)

            # Соматический выход при BAC-берсте усиливается нелинейным кальциевым гейтом:
            burst_gain = 1.65 if bac_burst_flag else 1.0
            u_soma = F.normalize(v_basal + 0.45 * burst_gain * apical_prediction_error + 1e-8, p=2, dim=-1)

            # Локальное правило Urbanczik-Senn + обучение SST-интернейронов:
            urbanczik_senn_delta = u_soma - v_basal
            if learn_sst:
                self.w_sst_pred.add_(0.05 * torch.outer(v_basal, apical_prediction_error))

            return {
                "u_soma": u_soma,
                "v_basal": v_basal,
                "apical_prediction_error": apical_prediction_error,
                "apical_prediction_error_norm": round(float(apical_prediction_error.norm().item()), 4),
                "mg_unblock_factor": round(mg_unblock, 4),
                "bac_burst_flag": bac_burst_flag,
                "burst_gain": burst_gain,
                "urbanczik_senn_error_norm": round(float(urbanczik_senn_delta.norm().item()), 4),
            }


# ============================================================================
# ПРОРЫВ 3 (ЭТАП 4): КАСКАДНАЯ СИНАПТИЧЕСКАЯ МЕТАПЛАСТИЧНОСТЬ ФУЗИ–ЗЕНКЕ (`Metaplasticity Shield`)
# ============================================================================
class FusiZenkeCascadeMetaplasticityShield:
    """Медленная трасса метапластичности: защита от перезаписи ПО ИСПОЛЬЗОВАНИЮ.

    ЧТО БЫЛО (семантика `magnitude_v1`, архив 2026-11-13). Трасса `u_slow`
    копила ВЕЛИЧИНУ веса (`u_slow += λ·|w|`), и затухание обновления
    `1/(1+κ·u_slow)` было тем сильнее, чем крупнее вес. Замеренные
    последствия: 34.5 % строк с attenuation < 1e-3, и у них МЕДИАННЫЙ |w|
    ВЫШЕ, чем у живых (1.597 против 1.214) — щит замораживал именно
    накопленный смысл. Механизм доказан по коду: трасса росла в
    `consolidate_into_slow_reservoir` (вызывается во сне) для КАЖДОГО
    элемента с |w| > 0.05, то есть каждая волна сна замораживала все
    содержательные строки всё сильнее, без предела.

    ЧТО СТАЛО (семантика `reinforcement_v2`). Трасса — МЕДЛЕННАЯ ИСТОРИЯ
    ПОДКРЕПЛЕНИЯ, а не величина веса:

      * раскладка — вектор на строку (`d_dend`, 32 КБ вместо матрицы
        8192x11377 = 372.8 МБ в чекпоинте и столько же в RAM);
      * прирост трассы задаётся ИСПОЛЬЗОВАНИЕМ: событие пластичности
        добавляет `λ·(средняя |Δ| по активным колонкам)`, причём по
        ПРИМЕНЁННОЙ дельте (затухание входит в неё) — поэтому замороженная
        строка не «докармливает» трассу и может decays-ом уйти обратно в
        пластичность;
      * затухание обновления считается по ПРЕДШЕСТВУЮЩЕЙ трассе
        (`1/(1+κ·B_prior)`), а не по трассе, в которую только что вписали
        это же событие: событие не гасит само себя (в старой версии
        `apply_sparse_update` трассу вообще не трогал, но `protect_rows`
        и сон добавляли `λ·|w|` каждый раз);
      * трасса ограничена сверху (`trace_cap`), поэтому затухание никогда
        не опускается ниже `1/(1+κ·trace_cap)` — обучение не может умереть;
      * во сне трасса глохнет (`sleep_decay`): неиспользуемые строки
        теряют защиту («use it or lose it»), используемые — подкрепляются
        событиями и остаются консолидированными.

    Миграция состояния: чекпоинты со старой семантикой (матрица
    `magnitude_v1`) переводятся в вектор и ОБНУЛЯЮТСЯ — старая трасса
    копила величину веса, в новой семантике она означала бы «заморожено
    по старому правилу». Обнуление — это и есть отморозка ошибочно
    замороженного. Факт миграции возвращается из `load_state_dict`.
    """

    SEMANTICS = "reinforcement_v2"
    LAYOUT = "row_trace"

    def __init__(self, d_dend: int = 8192, n_cols: int = 0, lambda_cons: float = 0.85,
                 delta_decay: float = 0.002, kappa_shield: float = 18.0,
                 trace_cap: float = 4.0, sleep_decay: float = 0.10,
                 protect_credit: float = 0.05):
        if (d_dend <= 0 or n_cols < 0 or lambda_cons < 0 or not 0 <= delta_decay <= 1
                or kappa_shield < 0 or trace_cap <= 0 or not 0 <= sleep_decay <= 1
                or protect_credit < 0
                or not all(math.isfinite(v) for v in (lambda_cons, delta_decay, kappa_shield,
                                                       trace_cap, sleep_decay, protect_credit))):
            raise ValueError("invalid shield configuration")
        self.d_dend, self.n_cols = int(d_dend), int(n_cols)
        self.lambda_cons = float(lambda_cons)
        self.delta_decay = float(delta_decay)
        self.kappa_shield = float(kappa_shield)
        self.trace_cap = float(trace_cap)
        self.sleep_decay = float(sleep_decay)
        self.protect_credit = float(protect_credit)
        # Ленивая трасса: (d_dend,) — до первого использования пуста.
        self.u_slow = torch.zeros(0, dtype=torch.float32)
        self.u_slow_semantics = self.SEMANTICS
        self.last_migration: Dict[str, Any] = {}

    # ── служебное ───────────────────────────────────────────────────────────
    def _normalize_trace(self) -> None:
        """Привести трассу к актуальной раскладке (миграция состояния).

        Лениво, потому что восстановление идёт общим путём
        `load_runtime_tree` (атрибуты ставятся напрямую, `load_state_dict`
        щита не вызывается). Старые чекпоинты несут МАТРИЦУ (d_dend, n_cols)
        с историей ВЕЛИЧИНЫ веса; в новой семантике такая трасса означала бы
        «заморожено по старому правилу», поэтому она сбрасывается, а
        раскладка становится вектором на строку. Факт миграции пишется в
        `last_migration`.
        """
        raw = self.u_slow
        if raw.ndim == 2:
            legacy_shape = [int(s) for s in raw.shape]
            self.u_slow = torch.zeros((self.d_dend,), dtype=raw.dtype, device=raw.device)
            self.u_slow_semantics = self.SEMANTICS
            self.last_migration = {"u_slow_semantics_from": str(self.last_migration.get(
                "u_slow_semantics_from", "magnitude_v1")),
                "u_slow_semantics_to": self.SEMANTICS,
                "u_slow_layout_from": "matrix", "u_slow_layout_to": self.LAYOUT,
                "u_slow_shape_from": legacy_shape, "u_slow_shape_to": [self.d_dend],
                "u_slow_reset": True}
            return
        if raw.ndim == 1 and raw.numel() != self.d_dend:
            grown = torch.zeros((self.d_dend,), dtype=raw.dtype, device=raw.device)
            keep = min(int(raw.numel()), self.d_dend)
            if keep:
                grown[:keep].copy_(raw[:keep])
            self.u_slow = grown
        if str(self.u_slow_semantics) != self.SEMANTICS:
            self.u_slow.zero_()
            self.u_slow_semantics = self.SEMANTICS
            self.last_migration = {"u_slow_semantics_from": str(self.last_migration.get(
                "u_slow_semantics_from", "magnitude_v1")),
                "u_slow_semantics_to": self.SEMANTICS,
                "u_slow_layout_from": self.last_migration.get("u_slow_layout_from", "row_trace"),
                "u_slow_layout_to": self.LAYOUT,
                "u_slow_shape_from": [int(self.u_slow.numel())],
                "u_slow_shape_to": [int(self.u_slow.numel())],
                "u_slow_reset": True}

    def _ensure(self, w_fast: torch.Tensor, active_cols: Optional[int]) -> int:
        if w_fast.ndim != 2 or not w_fast.is_floating_point():
            raise ValueError("w_fast must be a floating matrix")
        n = w_fast.shape[1] if active_cols is None else int(active_cols)
        if not 0 <= n <= w_fast.shape[1]:
            raise ValueError("active_cols outside w_fast")
        if self.u_slow.numel() == 0:
            self.d_dend = int(w_fast.shape[0])
        elif self.d_dend != int(w_fast.shape[0]):
            raise ValueError("reservoir row count does not match w_fast")
        # СНАЧАЛА миграция (legacy-матрица → вектор со сбросом), потом аллокация:
        # иначе матрица схлопнулась бы в первую строку вместо сброса.
        self._normalize_trace()
        self.u_slow = self.u_slow.to(device=w_fast.device, dtype=w_fast.dtype)
        if self.u_slow.numel() != self.d_dend:
            grown = w_fast.new_zeros((self.d_dend,))
            keep = min(int(self.u_slow.numel()), self.d_dend)
            if keep:
                grown[:keep].copy_(self.u_slow[:keep])
            self.u_slow = grown
        self.n_cols = max(int(self.n_cols), n)
        return n

    @staticmethod
    def _rows(row_indices, w_fast):
        rows = torch.as_tensor(row_indices, device=w_fast.device)
        if rows.ndim != 1 or (rows.numel() and rows.dtype not in (torch.int32, torch.int64)):
            raise ValueError("row_indices must be a one-dimensional integer sequence")
        rows = rows.long()
        if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= w_fast.shape[0]):
            raise ValueError("row index outside w_fast")
        return rows

    def _attenuation(self, slow: torch.Tensor) -> torch.Tensor:
        """Множитель обновления по ПРЕДШЕСТВУЮЩЕЙ трассе (1/(1+κ·B))."""
        return 1.0 / (1.0 + self.kappa_shield * slow)

    def _reinforcement(self, delta: torch.Tensor) -> torch.Tensor:
        """Подкрепление события на строку: средняя |Δ| по активным колонкам.

        Единицы — «вес на колонку», те же, что у |w|; величина веса в
        подкреплении НЕ участвует (иначе щит замораживал бы крупное).
        """
        if delta.shape[1] == 0:
            return torch.zeros(delta.shape[0], device=delta.device, dtype=delta.dtype)
        return delta.abs().sum(dim=1) / delta.shape[1]

    def _store_trace(self, rows: torch.Tensor, added: torch.Tensor) -> None:
        """Записать прирост трассы (с затуханием и потолком) для строк."""
        slow = self.u_slow.index_select(0, rows)
        updated = (slow * (1.0 - self.delta_decay) + added).clamp_(0.0, self.trace_cap)
        self.u_slow.index_copy_(0, rows, updated)

    # ── измерение ──────────────────────────────────────────────────────────
    def stats(self) -> Dict[str, Any]:
        """Сводка по трассе: затухание, доля замороженных, покрытие."""
        self._normalize_trace()
        if self.u_slow.numel() == 0:
            return {"rows": 0, "trace_mean": 0.0, "trace_max": 0.0, "trace_min": 0.0,
                    "attenuation_mean": 1.0, "attenuation_min": 1.0,
                    "attenuation_median": 1.0, "frozen_rows_lt_1e3": 0,
                    "protected_rows_lt_1e2": 0, "attenuation_floor": float(
                        1.0 / (1.0 + self.kappa_shield * self.trace_cap))}
        att = self._attenuation(self.u_slow)
        return {"rows": int(self.u_slow.numel()),
                "trace_mean": round(float(self.u_slow.mean().item()), 6),
                "trace_max": round(float(self.u_slow.max().item()), 6),
                "trace_min": round(float(self.u_slow.min().item()), 6),
                "attenuation_mean": round(float(att.mean().item()), 6),
                "attenuation_min": round(float(att.min().item()), 6),
                "attenuation_median": round(float(att.median().item()), 6),
                "frozen_rows_lt_1e3": int((att < 1e-3).sum().item()),
                "protected_rows_lt_1e2": int((att < 1e-2).sum().item()),
                "attenuation_floor": round(float(1.0 / (1.0 + self.kappa_shield * self.trace_cap)), 6)}

    # ── консолидация (сон) ─────────────────────────────────────────────────
    def consolidate_into_slow_reservoir(self, w_fast: torch.Tensor, active_cols=None,
                                        theta_consolidation=1) -> float:
        """Событие консолидации во сне: трасса глохнет, измеряется покрытие.

        Возвращает долю активных элементов веса, несущих консолидированное
        содержание (|w| > 0.05), в процентах. В отличие от старой версии
        величина веса здесь НЕ попадает в трассу: консолидация измеряется,
        но не управляет обучением.
        """
        n = self._ensure(w_fast, active_cols)
        theta = float(theta_consolidation)
        if not math.isfinite(theta) or theta < 0:
            raise ValueError("invalid consolidation strength")
        if not n or not w_fast.shape[0]:
            return 0.0
        with torch.no_grad():
            decay = 1.0 - (1.0 - self.sleep_decay) ** max(1.0, theta)
            self.u_slow.mul_(1.0 - decay)
            magnitude = w_fast[:, :n].abs()
            consolidated = (magnitude > 0.05) & (magnitude > 0.0)
            return float(consolidated.float().mean().item() * 100)

    def decay_trace(self, factor: Optional[float] = None) -> Dict[str, Any]:
        """Глохнуть трассу (забывание защиты неиспользуемых строк)."""
        self._ensure_trace_only()
        rate = self.sleep_decay if factor is None else float(factor)
        if not 0.0 <= rate <= 1.0:
            raise ValueError("decay factor must be within [0, 1]")
        before = self.stats()
        with torch.no_grad():
            self.u_slow.mul_(1.0 - rate)
        after = self.stats()
        return {"decay": rate, "before": before, "after": after}

    def _ensure_trace_only(self) -> None:
        if self.u_slow.numel() != self.d_dend:
            self.u_slow = torch.zeros((self.d_dend,), dtype=self.u_slow.dtype,
                                      device=self.u_slow.device)

    # ── пластичность ───────────────────────────────────────────────────────
    def protect_rows(self, w_fast: torch.Tensor, row_indices, active_cols,
                     credit: Optional[float] = None) -> float:
        """Пометить строки как консолидированные (поднять их трассу).

        Кредит задаётся по подкреплению, а не по величине веса.
        """
        rows = self._rows(row_indices, w_fast).unique()
        n = self._ensure(w_fast, active_cols)
        if not n or not rows.numel():
            return 0.0
        amount = self.protect_credit if credit is None else float(credit)
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("invalid protection credit")
        with torch.no_grad():
            self._store_trace(rows, torch.full((rows.numel(),), self.lambda_cons * amount,
                                               device=self.u_slow.device, dtype=self.u_slow.dtype))
            slow = self.u_slow.index_select(0, rows)
            return float((slow > 0.10).float().mean().item() * 100)

    def apply_metaplasticity_shielded_update(self, w_fast: torch.Tensor, delta_w: torch.Tensor,
                                             active_cols: int) -> None:
        n = self._ensure(w_fast, active_cols)
        if delta_w.ndim != 2 or delta_w.shape[0] != w_fast.shape[0] or delta_w.shape[1] < n:
            raise ValueError("delta_w must cover the active matrix slice")
        with torch.no_grad():
            delta = delta_w[:, :n].to(device=w_fast.device, dtype=w_fast.dtype)
            attenuation = self._attenuation(self.u_slow)
            applied = delta * attenuation.unsqueeze(1)
            w_fast[:, :n].add_(applied)
            self._store_trace(torch.arange(w_fast.shape[0], device=w_fast.device),
                              self.lambda_cons * self._reinforcement(applied))

    def apply_sparse_update(self, w_fast: torch.Tensor, row_indices, delta_rows: torch.Tensor,
                            active_cols) -> None:
        rows = self._rows(row_indices, w_fast)
        n = self._ensure(w_fast, active_cols)
        if delta_rows.ndim != 2 or delta_rows.shape[0] != rows.numel() or delta_rows.shape[1] < n:
            raise ValueError("delta_rows must cover only the gathered rows and active columns")
        with torch.no_grad():
            delta = delta_rows[:, :n].to(device=w_fast.device, dtype=w_fast.dtype)
            # Затухание по ПРЕДШЕСТВУЮЩЕЙ трассе: событие не гасит себя.
            attenuation = self._attenuation(self.u_slow.index_select(0, rows))
            applied = delta * attenuation.unsqueeze(1)
            # index_add handles repeated row indices without losing any updates.
            w_fast[:, :n].index_add_(0, rows, applied)
            # История подкрепления — по ПРИМЕНЁННОЙ дельте, суммарно по строке.
            uniq, inverse = torch.unique(rows, return_inverse=True)
            reward = torch.zeros(uniq.numel(), device=delta.device, dtype=delta.dtype)
            reward.index_add_(0, inverse, self._reinforcement(applied))
            self._store_trace(uniq, self.lambda_cons * reward)

    # ── состояние ──────────────────────────────────────────────────────────
    def export_state_dict(self) -> dict:
        return {"d_dend": self.d_dend, "n_cols": self.n_cols, "lambda_cons": self.lambda_cons,
                "delta_decay": self.delta_decay, "kappa_shield": self.kappa_shield,
                "trace_cap": self.trace_cap, "sleep_decay": self.sleep_decay,
                "protect_credit": self.protect_credit,
                "u_slow": self.u_slow.detach().cpu().clone(),
                "u_slow_semantics": self.u_slow_semantics,
                "u_slow_layout": self.LAYOUT}

    def load_state_dict(self, state: dict) -> dict:
        raw = torch.as_tensor(state["u_slow"]).detach().cpu().clone()
        if not raw.is_floating_point() or not (torch.isfinite(raw).all() and (raw >= 0).all()):
            raise ValueError("invalid shield reservoir")
        semantics = str(state.get("u_slow_semantics", "magnitude_v1"))
        layout = str(state.get("u_slow_layout", "matrix" if raw.ndim == 2 else self.LAYOUT))
        migration = {"u_slow_semantics_from": semantics, "u_slow_semantics_to": self.SEMANTICS,
                     "u_slow_layout_from": layout, "u_slow_layout_to": self.LAYOUT,
                     "u_slow_reset": False, "u_slow_shape_from": list(raw.shape)}
        d_dend = int(state["d_dend"])
        if raw.ndim == 2:
            # Старая раскладка: матрица (d_dend, n_cols) с историей ВЕЛИЧИНЫ.
            migration["u_slow_reset"] = True
            slow = torch.zeros((d_dend,), dtype=raw.dtype)
        elif raw.ndim == 1:
            slow = raw
            if semantics != self.SEMANTICS or layout != self.LAYOUT:
                # Старая семантика при новой раскладке — тоже сброс.
                migration["u_slow_reset"] = True
                slow = torch.zeros_like(slow)
        else:
            raise ValueError("invalid shield reservoir")
        candidate = type(self)(d_dend, int(state["n_cols"]), float(state["lambda_cons"]),
                               float(state["delta_decay"]), float(state["kappa_shield"]),
                               float(state.get("trace_cap", self.trace_cap)),
                               float(state.get("sleep_decay", self.sleep_decay)),
                               float(state.get("protect_credit", self.protect_credit)))
        candidate.u_slow = slow
        candidate.n_cols = int(state["n_cols"])
        candidate.u_slow_semantics = self.SEMANTICS
        candidate.u_slow_layout = self.LAYOUT
        migration["u_slow_shape_to"] = list(slow.shape)
        self.__dict__.update(candidate.__dict__)
        self.last_migration = migration
        return migration




# ============================================================================
# ПРОРЫВ 4 & 5: МЕЖДОМЕННЫЙ СИНТЕЗ ИЗОМОРФИЗМОВ (`REM Cross-Domain Dreamer`) & ДАШБОРД СОЗНАНИЯ
# ============================================================================
class CrossDomainIsomorphismDreamer:
    """Bounded exact alignment of named, directed, typed graphs; no keyword knowledge."""

    EDGE_LABELS = ("relation", "polarity", "causal_kind", "type")

    def __init__(self, distillery: Optional[KnowledgeDistillery] = None, max_search_states: int = 100000):
        self.distillery = distillery
        self.graphs: Dict[str, dict] = {}
        self.max_search_states = int(max_search_states)
        if self.max_search_states <= 0:
            raise ValueError("max_search_states must be positive")

    def register_graph(self, name: str, edges: Sequence[dict], node_types: dict[str, str]) -> None:
        if not isinstance(name, str) or not name.strip() or not node_types or len(node_types) > 64:
            raise ValueError("a named graph must have 1..64 typed nodes")
        if any(not isinstance(n, str) or not isinstance(t, str) or not n or not t for n, t in node_types.items()):
            raise ValueError("node names and types must be nonempty strings")
        normalized = []
        for item in edges:
            subject = item.get("subject", item.get("source"))
            obj = item.get("object", item.get("target"))
            direction = item.get("direction", "forward")
            if item.get("directed", True) is not True:
                raise ValueError("only directed edges are supported")
            if direction in ("reverse", "backward", "<-"):
                subject, obj = obj, subject
            elif direction not in ("forward", "->"):
                raise ValueError("only directed edges are supported")
            if subject not in node_types or obj not in node_types or not isinstance(item.get("relation"), str) or not item["relation"]:
                raise ValueError("each edge needs typed endpoints and a relation")
            polarity = str(item.get("polarity", "positive")).lower()
            polarity = {"1": "positive", "+1": "positive", "+": "positive", "-1": "negative", "-": "negative"}.get(polarity, polarity)
            normalized.append({"subject": subject, "object": obj, "relation": item["relation"],
                               "polarity": polarity, "causal_kind": str(item.get("causal_kind", "unspecified")),
                               "type": str(item.get("type", "unspecified"))})
        self.graphs[name] = {"edges": normalized, "node_types": dict(node_types)}

    def _align(self, source: dict, target: dict, heldout: Optional[dict] = None):
        st, tt = source["node_types"], target["node_types"]
        if len(st) != len(tt) or Counter(st.values()) != Counter(tt.values()):
            return [], False
        source_edges = list(source["edges"])
        if heldout is not None:
            source_edges.remove(heldout)
        # No evidence means no structural analogy, even if node types happen to match.
        if not source_edges:
            return [], False
        sa, ta = {}, {}
        for edges, adjacency in ((source_edges, sa), (target["edges"], ta)):
            for edge in edges:
                label = tuple(edge[k] for k in self.EDGE_LABELS)
                adjacency.setdefault((edge["subject"], edge["object"]), Counter())[label] += 1
        candidates = {s: [t for t in tt if tt[t] == st[s]] for s in st}
        degree = Counter(n for e in source_edges for n in (e["subject"], e["object"]))
        order = sorted(st, key=lambda n: (len(candidates[n]), -degree[n], n))
        solutions, mapping, used = [], {}, set()
        searched, exhausted = 0, False

        def pair_ok(a, b):
            expected = sa.get((a, b), Counter())
            actual = ta.get((mapping[a], mapping[b]), Counter()).copy()
            if heldout is not None and (a, b) == (heldout["subject"], heldout["object"]):
                # Mask all target versions of the queried relation: neither existence nor
                # its label attributes are allowed to select the alignment.
                actual = Counter({label: count for label, count in actual.items() if label[0] != heldout["relation"]})
            return expected == actual

        def visit(depth):
            nonlocal searched, exhausted
            if len(solutions) >= 2 or exhausted:
                return
            if depth == len(order):
                solutions.append(dict(mapping))
                return
            s = order[depth]
            for t in candidates[s]:
                searched += 1
                if searched > self.max_search_states:
                    exhausted = True
                    return
                if t in used:
                    continue
                mapping[s] = t
                if all(pair_ok(s, a) and pair_ok(a, s) for a in mapping):
                    used.add(t)
                    visit(depth + 1)
                    used.remove(t)
                del mapping[s]

        visit(0)
        return ([] if exhausted else solutions), exhausted

    @staticmethod
    def _proof(edges, mapping):
        return [{"source_edge": dict(e), "target_edge": {
            **e, "subject": mapping[e["subject"]], "object": mapping[e["object"]]
        }} for e in edges]

    def match_graphs(self, source: str, target: str) -> Optional[dict]:
        if source not in self.graphs or target not in self.graphs:
            return None
        mappings, _ = self._align(self.graphs[source], self.graphs[target])
        if not mappings:
            return None
        mapping = mappings[0]
        return {"mapping": mapping, "score": 1.0, "ambiguous": len(mappings) > 1,
                "proof": self._proof(self.graphs[source]["edges"], mapping)}

    def transfer_relation(self, source, target, subject, relation, object) -> Optional[dict]:
        if source not in self.graphs or target not in self.graphs:
            return None
        sg, tg = self.graphs[source], self.graphs[target]
        queried = [e for e in sg["edges"] if (e["subject"], e["relation"], e["object"]) == (subject, relation, object)]
        if len(queried) != 1:
            return None
        heldout = queried[0]
        # Another parallel copy would leak the held-out relation into alignment.
        if any(e is not heldout and (e["subject"], e["object"], e["relation"]) == (subject, object, relation) for e in sg["edges"]):
            return None
        mappings, _ = self._align(sg, tg, heldout)
        if len(mappings) != 1:
            return None  # Ambiguous analogies are not predictions.
        mapping = mappings[0]
        transferred = {**heldout, "subject": mapping[subject], "object": mapping[object]}
        observed = [e for e in tg["edges"] if (e["subject"], e["object"], e["relation"]) == (
            transferred["subject"], transferred["object"], relation
        )]
        if any(e != transferred for e in observed):
            return None
        evidence = list(sg["edges"])
        evidence.remove(heldout)
        return {**transferred, "edge": transferred, "mapping": mapping, "score": 1.0,
                "proof": self._proof(evidence, mapping), "heldout_edge": dict(heldout),
                "status": "already_observed" if observed else "structural_prediction"}

    def resolve_cross_domain_isomorphism(self, utterance: str) -> Optional[dict]:
        if not re.search(r"\b(compare|comparison|analogy|analogous|isomorphism)\b", utterance, re.I):
            return None
        names = [name for name in self.graphs if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", utterance, re.I)]
        if len(names) != 2:
            return None
        result = self.match_graphs(*names)
        if result is not None:
            result["domains_bridged"] = names
        return result

    def export_state_dict(self) -> dict:
        return {"graphs": copy.deepcopy(self.graphs), "max_search_states": self.max_search_states}

    def load_state_dict(self, state: dict) -> None:
        candidate = type(self)(self.distillery, int(state["max_search_states"]))
        for name, graph in state["graphs"].items():
            candidate.register_graph(name, graph["edges"], graph["node_types"])
        self.graphs, self.max_search_states = candidate.graphs, candidate.max_search_states


class ConsciousnessMetricsDashboard:
    """Measured trace diagnostics, not IIT estimates or consciousness certification."""

    @staticmethod
    def _array(value):
        if value is None:
            return None
        try:
            array = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64)
            return array if array.numel() and torch.isfinite(array).all() else None
        except (TypeError, ValueError, RuntimeError):
            return None

    def compute_turn_consciousness_metrics(
        self, ca3_resonance=None, epistemic_uncertainty=None,
        cortical_trajectory_trace: Sequence[Sequence[str]] = (), semantic_closure_score=0.90,
        *, spike_raster=None, perturbed_spike_raster=None, basal_energy=None, apical_energy=None,
        confidence_outcomes=None, attention_predictions=None, attention_targets=None, total_columns=None,
    ) -> dict:
        # Legacy resonance/uncertainty/closure arguments are not substitutes for measurements.
        result = dict.fromkeys(("gwt_ignition_score", "broadcast_reach", "iit_pci_complexity_proxy",
                               "rpt_recurrence_index", "hot_metacognitive_confidence", "hot_calibration_brier",
                               "hot_calibration_score", "ast_attention_schema_alignment", "priesemann_subcritical_sigma"))
        raster = self._array(spike_raster)
        perturbed = self._array(perturbed_spike_raster)
        if raster is not None and (raster.ndim != 2 or (raster < 0).any()):
            raster = None
        if raster is not None:
            n_cols = int(total_columns) if total_columns is not None else raster.shape[1]
            if n_cols >= raster.shape[1] and n_cols > 0:
                result["gwt_ignition_score"] = float((raster > 0).sum(dim=1).max() / n_cols)
                result["broadcast_reach"] = float((raster > 0).any(dim=0).sum() / n_cols)
            activity = raster.sum(dim=1)
            if activity.numel() > 1:
                valid = activity[:-1] > 0
                denominator = activity[:-1][valid].sum()
                if denominator > 0:
                    # Observational activity ratio, not a causal branching-process estimator.
                    result["priesemann_subcritical_sigma"] = float(activity[1:][valid].sum() / denominator)
            if perturbed is not None and perturbed.shape == raster.shape and (perturbed >= 0).all() and raster.shape[0] > 1:
                response = perturbed != raster
                if response.any():
                    raw = bytes(response.to(torch.uint8).reshape(-1).tolist())
                    compressor = zlib.compressobj(wbits=-15)
                    compressed = compressor.compress(raw) + compressor.flush()
                    result["iit_pci_complexity_proxy"] = len(compressed) / len(raw)
                else:
                    result["iit_pci_complexity_proxy"] = 0.0
        elif cortical_trajectory_trace and total_columns is not None and int(total_columns) > 0:
            sets = [set(step) for step in cortical_trajectory_trace]
            n_cols = int(total_columns)
            unique = set().union(*sets)
            if len(unique) <= n_cols:
                result["gwt_ignition_score"] = max(map(len, sets)) / n_cols
                result["broadcast_reach"] = len(unique) / n_cols

        basal, apical = self._array(basal_energy), self._array(apical_energy)
        if basal is not None and apical is not None and basal.shape == apical.shape and (basal >= 0).all() and (apical >= 0).all():
            total = basal.sum() + apical.sum()
            if total > 0:
                result["rpt_recurrence_index"] = float(apical.sum() / total)

        outcomes = self._array(confidence_outcomes)
        if outcomes is not None and outcomes.ndim == 2 and outcomes.shape[1] == 2:
            confidence, observed = outcomes[:, 0], outcomes[:, 1]
            if ((confidence >= 0) & (confidence <= 1)).all() and ((observed == 0) | (observed == 1)).all():
                brier = float(((confidence - observed) ** 2).mean())
                result["hot_metacognitive_confidence"] = float(confidence.mean())
                result["hot_calibration_brier"] = brier
                result["hot_calibration_score"] = 1 - brier

        predictions, targets = self._array(attention_predictions), self._array(attention_targets)
        if predictions is not None and targets is not None:
            if predictions.ndim == 2:
                predictions = predictions.argmax(dim=1)
            if targets.ndim == 2:
                targets = targets.argmax(dim=1)
            if predictions.ndim == targets.ndim == 1 and predictions.shape == targets.shape:
                result["ast_attention_schema_alignment"] = float((predictions == targets).double().mean())
        result["unavailable_measurements"] = [key for key, value in result.items() if value is None]
        return result


# ============================================================================
# EVOTRAIN v25.1 ИННОВАЦИЯ 1: 16-БАНКОВОЕ КОРТИКАЛЬНОЕ РАЗДЕЛЕНИЕ (`CorticalColumnarSubBankRouter`)
# ============================================================================
class CorticalColumnarSubBankRouter:
    """Sparse learned residual banks split on observed cross-pole gradient conflict.

    The budget uses conservative state-entry accounting (not process RSS or temporary
    tensors). Saturated banks/budgets reject conflicting updates rather than evicting
    learned state. On acceptance, entries with |weight| <= prune_threshold are pruned
    (negligible residual); the pruned candidate must still not worsen the original loss.
    Candidate logits are residuals for the caller's shared readout.
    """

    def __init__(self, n_banks: int = 16, d_vec: int = 64, seed: int = 2501,
                 memory_budget_bytes: int = 8 * 1024 * 1024, max_active_features: int = 128,
                 max_token_ids: int = 256, conflict_threshold: float = -0.05,
                 prune_threshold: float = 1e-4):
        self.n_banks, self.d_vec, self.seed = int(n_banks), int(d_vec), int(seed)
        self.memory_budget_bytes = int(memory_budget_bytes)
        self.max_active_features, self.max_token_ids = int(max_active_features), int(max_token_ids)
        self.conflict_threshold = float(conflict_threshold)
        self.prune_threshold = float(prune_threshold)
        if not 1 <= self.n_banks <= 256 or self.d_vec <= 0 or min(self.max_active_features, self.max_token_ids) <= 0 or not -1 <= self.conflict_threshold < 0:
            raise ValueError("invalid bank configuration")
        if not math.isfinite(self.prune_threshold) or self.prune_threshold < 0:
            raise ValueError("prune_threshold must be nonnegative and finite")
        self.banks: List[dict] = [{} for _ in range(self.n_banks)]
        self.domains: Dict[str, dict] = {}
        self.bank_pole_counts = {i: 0 for i in range(self.n_banks)}
        if self.memory_usage_bytes > self.memory_budget_bytes:
            raise ValueError("memory budget cannot hold the empty banks")

    def _memory_usage(self, banks, domains):
        # Includes key/value/container allowance; avoids unbounded domain/gradient history.
        return (128 * self.n_banks + sum(192 * len(b) for b in banks)
                + sum(512 + 4 * self.d_vec + 192 * len(d["gradient"]) for d in domains.values()))

    @property
    def memory_usage_bytes(self) -> int:
        return self._memory_usage(self.banks, self.domains)

    @property
    def allocated_banks(self) -> int:
        return len({d["bank"] for d in self.domains.values()})

    @property
    def bank_basis(self) -> torch.Tensor:
        basis = torch.zeros((self.n_banks, self.d_vec))
        for domain in self.domains.values():
            basis[domain["bank"]].add_(domain["pole"])
        return F.normalize(basis, dim=1)

    def _pole(self, pole_vec64):
        pole = pole_vec64.detach().to(device="cpu", dtype=torch.float32)
        if pole.shape != (self.d_vec,) or not torch.isfinite(pole).all() or not pole.norm() > 0:
            raise ValueError("expected a finite nonzero pole vector")
        pole = F.normalize(pole, dim=0)
        key = hashlib.sha256(bytes(pole.contiguous().view(torch.uint8).tolist())).hexdigest()
        return key, pole

    def _features(self, features: torch.Tensor) -> dict:
        if features.ndim != 1 or not features.is_floating_point():
            raise ValueError("features must be a one-dimensional floating tensor")
        if features.layout == torch.sparse_coo:
            sparse = features.detach().coalesce()
            indices, values = sparse.indices()[0], sparse.values()
        elif features.layout == torch.strided:
            indices = features.detach().nonzero(as_tuple=True)[0]
            values = features.detach().index_select(0, indices)
        else:
            raise ValueError("use a dense vector or sparse COO features")
        if not torch.isfinite(values).all():
            raise ValueError("features must be finite")
        if values.numel() > self.max_active_features:
            selected = values.abs().topk(self.max_active_features, sorted=False).indices
            indices, values = indices[selected], values[selected]
        return {int(i): float(v) for i, v in zip(indices.cpu().tolist(), values.cpu().tolist()) if v != 0}

    def _tokens(self, token_ids, *, update_mask=False):
        values = list(token_ids)
        tokens = [int(t) for t in values]
        if (len(tokens) != len(set(tokens)) or any(t < 0 or t != value for t, value in zip(tokens, values))
                or (update_mask and len(tokens) > self.max_token_ids)):
            raise ValueError("tokens must be unique, nonnegative integers; update masks must fit the token budget")
        return tokens

    def route_pole_to_sub_bank(self, pole_vec64: torch.Tensor) -> int:
        key, pole = self._pole(pole_vec64)
        if key in self.domains:
            return self.domains[key]["bank"]
        if not self.domains:
            return 0
        nearest = max(self.domains.values(), key=lambda d: float(torch.dot(pole, d["pole"])))
        return nearest["bank"]

    def compute_bank_distribution(self, ca3_attractors: torch.Tensor) -> Dict[int, int]:
        if ca3_attractors.ndim != 2 or ca3_attractors.shape[1] != self.d_vec:
            raise ValueError("expected a matrix of pole vectors")
        counts = {i: 0 for i in range(self.n_banks)}
        for pole in ca3_attractors:
            counts[self.route_pole_to_sub_bank(pole)] += 1
        self.bank_pole_counts = counts
        return dict(counts)

    @staticmethod
    def _scores(bank, features, tokens):
        positions = {token: col for col, token in enumerate(tokens)}
        active_entries = {}
        for (idx, token), weight in bank.items():
            if idx in features and token in positions:
                active_entries.setdefault(idx, []).append((positions[token], weight))
        contributions = [[] for _ in tokens]
        for idx, value in features.items():
            for col, weight in active_entries.get(idx, ()):
                contributions[col].append(value * weight)
        # Preserve native compensated summation and feature order.
        return [sum(values) for values in contributions]

    def predict_sparse(self, pole_vec64, features: torch.Tensor, token_ids: Sequence[int]) -> torch.Tensor:
        tokens, sparse = self._tokens(token_ids), self._features(features)
        bank = self.banks[self.route_pole_to_sub_bank(pole_vec64)]
        return torch.tensor(self._scores(bank, sparse, tokens), device=features.device, dtype=features.dtype)

    def learn_sparse(self, pole_vec64, features: torch.Tensor, target_token: int,
                     token_ids: Sequence[int], learning_rate: float = 0.1, *,
                     all_token_ids: Sequence[int], base_logits: torch.Tensor, residual_gain: float) -> dict:
        tokens = self._tokens(token_ids, update_mask=True)
        all_tokens = self._tokens(all_token_ids)
        positions = {token: i for i, token in enumerate(all_tokens)}
        target_token, learning_rate = int(target_token), float(learning_rate)
        residual_gain = float(residual_gain)
        if target_token not in tokens or any(t not in positions for t in tokens):
            raise ValueError("update mask must contain the target and be a subset of the active vocabulary")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(residual_gain) or residual_gain <= 0:
            raise ValueError("residual_gain must be positive and finite")
        if (not isinstance(base_logits, torch.Tensor) or base_logits.ndim != 1
                or not base_logits.is_floating_point() or base_logits.layout != torch.strided
                or base_logits.numel() != len(all_tokens) or not torch.isfinite(base_logits).all()):
            raise ValueError("base_logits must be a finite floating vector aligned with the active vocabulary")
        base = base_logits.detach().to(device="cpu", dtype=torch.float64)
        sparse = self._features(features)
        key, pole = self._pole(pole_vec64)
        original_bank_id = bank_id = self.route_pole_to_sub_bank(pole_vec64)
        target_position = positions[target_token]

        def objective(bank, *, with_gradient=False):
            raw = torch.tensor(self._scores(bank, sparse, all_tokens), dtype=torch.float64)
            bounded = raw.tanh()
            scores = base + residual_gain * bounded
            loss = float(scores.logsumexp(dim=0) - scores[target_position])
            if not with_gradient:
                return loss
            errors = scores.softmax(dim=0)
            errors[target_position] -= 1
            derivatives = residual_gain * (1 - bounded.square()) * errors
            gradient = {(idx, token): value * float(derivatives[positions[token]])
                        for idx, value in sparse.items() for token in tokens
                        if derivatives[positions[token]] != 0}
            return loss, gradient

        def selection_stats(selected, gradient):
            return {"update_tokens_requested": len(tokens), "update_tokens_selected": len(selected),
                    "update_mask_limited": len(selected) < len(tokens),
                    "gradient_entries_selected": len(gradient)}

        def feasible_gradient(bank, full_gradient):
            used = self.memory_usage_bytes
            if self.prune_threshold > 0:
                used -= 192 * sum(1 for w in bank.values() if abs(w) <= self.prune_threshold)
            if key in self.domains:
                used -= 192 * len(self.domains[key]["gradient"])
            else:
                used += 512 + 4 * self.d_vec
            costs, energies = dict.fromkeys(tokens, 0), dict.fromkeys(tokens, 0.0)
            for entry, grad in full_gradient.items():
                token = entry[1]
                costs[token] += 192 + (192 if entry not in bank else 0)
                energies[token] += grad * grad
            ordered = [target_token] + sorted((t for t in tokens if t != target_token),
                                              key=lambda t: (-energies[t], t))
            selected = set()
            for token in ordered:
                if used + costs[token] > self.memory_budget_bytes:
                    if token == target_token:
                        return {}, set()
                    continue
                selected.add(token)
                used += costs[token]
            return {entry: grad for entry, grad in full_gradient.items() if entry[1] in selected}, selected

        loss, full_gradient = objective(self.banks[bank_id], with_gradient=True)
        result = {"accepted": False, "bank": original_bank_id, "allocated_branch": False,
                  "gradient_conflict": 0.0, "loss": loss, "loss_after": loss,
                  "backtracking_steps": 0, "learning_rate": 0.0, "pruned_entries": 0,
                  "memory_usage_bytes": self.memory_usage_bytes,
                  "objective": "full_vocabulary_bounded_residual", **selection_stats((), {})}
        if not full_gradient:
            return {**result, "reason": "no_learning_signal"}
        gradient, selected = feasible_gradient(self.banks[bank_id], full_gradient)
        if not selected:
            return {**result, "reason": "memory_budget_exhausted"}
        if not gradient:
            return {**result, "reason": "no_learning_signal"}
        if not math.isfinite(loss):
            return {**result, **selection_stats(selected, gradient), "reason": "objective_not_improved"}
        norm = math.sqrt(sum(g * g for g in gradient.values()))
        conflict = 0.0
        for other_key, domain in self.domains.items():
            if other_key == key or domain["bank"] != bank_id:
                continue
            other = domain["gradient"]
            other_norm = math.sqrt(sum(g * g for g in other.values()))
            if norm and other_norm:
                cosine = sum(g * other.get(k, 0) for k, g in gradient.items()) / (norm * other_norm)
                conflict = min(conflict, cosine)
        result["gradient_conflict"] = conflict
        split = conflict < self.conflict_threshold
        if split:
            occupied = {d["bank"] for d in self.domains.values()}
            free = next((b for b in range(self.n_banks) if b not in occupied), None)
            if free is None:
                return {**result, "reason": "bank_budget_exhausted"}
            bank_id = free
            # Independent branch gradient; acceptance still uses the ORIGINAL routed loss.
            _, full_gradient = objective(self.banks[bank_id], with_gradient=True)
            if not full_gradient:
                return {**result, "reason": "no_learning_signal"}
            gradient, selected = feasible_gradient(self.banks[bank_id], full_gradient)
            if not selected:
                return {**result, "reason": "memory_budget_exhausted"}
            if not gradient:
                return {**result, "reason": "no_learning_signal"}

        result.update(selection_stats(selected, gradient))
        updated_domain = {"pole": pole.clone(), "bank": bank_id, "gradient": gradient,
                          "updates": self.domains.get(key, {}).get("updates", 0) + 1}
        prospective_domains = {**self.domains, key: updated_domain}
        step = learning_rate
        for attempt in range(12):
            updated_bank = dict(self.banks[bank_id])
            for entry, grad in gradient.items():
                updated_bank[entry] = updated_bank.get(entry, 0.0) - step * grad
            if self.prune_threshold > 0:
                candidate_bank = {e: w for e, w in updated_bank.items() if abs(w) > self.prune_threshold}
            else:
                candidate_bank = updated_bank
            pruned = len(updated_bank) - len(candidate_bank)
            prospective_banks = list(self.banks)
            prospective_banks[bank_id] = candidate_bank
            loss_after = objective(candidate_bank)
            if math.isfinite(loss_after) and loss_after <= loss and all(math.isfinite(v) for v in candidate_bank.values()):
                self.banks, self.domains = prospective_banks, prospective_domains
                self.bank_pole_counts = {b: sum(d["bank"] == b for d in self.domains.values()) for b in range(self.n_banks)}
                return {**result, "accepted": True, "bank": bank_id, "allocated_branch": split,
                        "loss_after": loss_after, "backtracking_steps": attempt, "learning_rate": step,
                        "pruned_entries": pruned, "memory_usage_bytes": self.memory_usage_bytes}
            result["loss_after"] = loss_after
            step *= 0.5
        return {**result, "reason": "objective_not_improved", "backtracking_steps": 12}

    def export_state_dict(self) -> dict:
        return copy.deepcopy({"n_banks": self.n_banks, "d_vec": self.d_vec, "seed": self.seed,
                              "memory_budget_bytes": self.memory_budget_bytes,
                              "max_active_features": self.max_active_features, "max_token_ids": self.max_token_ids,
                              "conflict_threshold": self.conflict_threshold, "prune_threshold": self.prune_threshold,
                              "banks": self.banks, "domains": self.domains, "bank_pole_counts": self.bank_pole_counts})

    def load_state_dict(self, state: dict) -> None:
        candidate = type(self)(**{k: state[k] for k in ("n_banks", "d_vec", "seed", "memory_budget_bytes",
                                                      "max_active_features", "max_token_ids", "conflict_threshold")},
                               prune_threshold=state.get("prune_threshold", 1e-4))
        banks, domains = copy.deepcopy(state["banks"]), copy.deepcopy(state["domains"])
        if len(banks) != candidate.n_banks:
            raise ValueError("invalid bank count")
        for key, domain in domains.items():
            pole = torch.as_tensor(domain["pole"]).detach().to(device="cpu", dtype=torch.float32).clone()
            if pole.shape != (candidate.d_vec,) or not torch.isfinite(pole).all() or not pole.norm() > 0:
                raise ValueError("invalid domain pole")
            computed = hashlib.sha256(bytes(pole.contiguous().view(torch.uint8).tolist())).hexdigest()
            if computed != key or not 0 <= domain["bank"] < candidate.n_banks or domain["updates"] < 1:
                raise ValueError("invalid domain state")
            domain["pole"] = pole
        for entries in banks + [d["gradient"] for d in domains.values()]:
            if any(not isinstance(k, tuple) or len(k) != 2 or any(not isinstance(i, int) or i < 0 for i in k)
                   or not math.isfinite(v) for k, v in entries.items()):
                raise ValueError("invalid sparse bank entries")
        if candidate._memory_usage(banks, domains) > candidate.memory_budget_bytes:
            raise ValueError("state exceeds bank memory budget")
        candidate.banks, candidate.domains = banks, domains
        candidate.bank_pole_counts = {int(k): int(v) for k, v in state["bank_pole_counts"].items()}
        self.__dict__.update(candidate.__dict__)


# ============================================================================
# EVOTRAIN v25.1 ИННОВАЦИЯ 3: СОКРАТИЧЕСКИЙ СИНТЕЗАТОР ВОПРОСОВ (`BA46SocraticInverseQuerySynthesizer`)
# ============================================================================
class BA46SocraticInverseQuerySynthesizer:
    r"""Эндогенный Сократический Генератор Вопросов (`BA46 Inverse Query Synthesizer`, `EVOTRAIN v25.1 Innovation 3`):
      - Превращает декларативные научные утверждения из монографий `Knowledge/` и `WikiText-103`
        вида $\text{Subject} + \text{CausalVerb} + \text{Object}$ в естественные вопросительные полюса
        (`"How does <Subject> <BaseVerb> <Object>?"`), обеспечивая тройную запись каждого факта:
        1) Вопросно-ответный полюс в `CA3`;
        2) Эпизодическую цепь в `w_ca3_episode_links`;
        3) Каузальное ребро в `CorticalRelationalWorldGraph`."""

    CAUSAL_VERB_STEMS: Dict[str, Tuple[str, str]] = {
        "releases": ("does", "release"),
        "release": ("do", "release"),
        "activates": ("does", "activate"),
        "activate": ("do", "activate"),
        "inhibits": ("does", "inhibit"),
        "inhibit": ("do", "inhibit"),
        "suppresses": ("does", "suppress"),
        "suppress": ("do", "suppress"),
        "produces": ("does", "produce"),
        "produce": ("do", "produce"),
        "fuels": ("does", "fuel"),
        "fuel": ("do", "fuel"),
        "stimulates": ("does", "stimulate"),
        "stimulate": ("do", "stimulate"),
        "phosphorylates": ("does", "phosphorylate"),
        "phosphorylate": ("do", "phosphorylate"),
        "stabilizes": ("does", "stabilize"),
        "stabilize": ("do", "stabilize"),
        "absorbs": ("does", "absorb"),
        "absorb": ("do", "absorb"),
        "splits": ("does", "split"),
        "split": ("do", "split"),
        "directs": ("does", "direct"),
        "direct": ("do", "direct"),
        "shields": ("does", "shield"),
        "shield": ("do", "shield"),
    }

    def synthesize_socratic_question_from_fact(self, declarative_sentence: str) -> Optional[str]:
        clean = " ".join(declarative_sentence.strip().rstrip(".!?").split())
        words = clean.split()
        if len(words) < 5:
            return None
        for idx, w in enumerate(words[1:-1], start=1):
            w_low = w.lower()
            if w_low in self.CAUSAL_VERB_STEMS:
                aux, base_v = self.CAUSAL_VERB_STEMS[w_low]
                subj = " ".join(words[:idx])
                subj_low = subj[0].lower() + subj[1:] if len(subj) > 1 and not subj[:2].isupper() else subj
                obj = " ".join(words[idx + 1 :])
                return f"How {aux} {subj_low} {base_v} {obj}?"
        return None

