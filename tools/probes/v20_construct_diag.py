"""Локализация падения при построении интегрированного организма (runtime_version=20).

v20 — это «идеальный» контур (IntegratedOrganism). Прогон карты модулей на нём
упал с ACCESS VIOLATION (0xC0000005) через 0.3 с после аллокации 3.2 ГБ.
Здесь тот же сценарий с faulthandler, чтобы получить стек.
"""
from __future__ import annotations

import faulthandler
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
faulthandler.enable(all_threads=True)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

print(">>> строим организм runtime_version=20", flush=True)
from ziwago_assistant import ZiwagoBiologicalSNNAssistant  # noqa: E402

assistant = ZiwagoBiologicalSNNAssistant(
    root=ROOT, checkpoint_path=None, workers=2, cpu_threads=4, seed=42, runtime_version=20)
print(">>> построен OK", flush=True)
core = assistant.organism
print(">>> атрибуты интеграции:", [a for a in vars(core) if "integrat" in a.lower()
                                  or "critical" in a.lower() or "hippo" in a.lower()], flush=True)
for attr in ("integrated", "integrated_organism", "_integrated"):
    if hasattr(core, attr):
        print(f">>> {attr} =", type(getattr(core, attr)).__name__, flush=True)
print(">>> флаги:", {k: getattr(core, k, None) for k in (
    "criticality_control_enabled", "volume_chemistry_enabled", "interoception_enabled",
    "limbic_enabled", "dual_hippocampus_enabled")}, flush=True)

# Один настоящий шаг обучения — здесь v20-путь падал с ACCESS VIOLATION.
from ziwago.app.data import TokenizedDialogPair  # noqa: E402

ids = core.encode_text("I am making mayonnaise and it has become runny.")
pair = TokenizedDialogPair("probe", ids, ids, quality_reward_hint=2.0)
print(">>> шаг обучения v20 ...", flush=True)
stats = core.train_wave_with_neuromodulator_reward([pair], wave_idx=1)
print(">>> шаг обучения v20 OK:", {k: stats.get(k) for k in (
    "readout_token_updates", "dendritic_tokens_trained", "readout_step_limited")}, flush=True)
