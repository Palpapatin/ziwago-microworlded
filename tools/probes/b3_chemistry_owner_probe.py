"""B.3: Единый владелец химии — доказательство замером (лёгкий tiny_core, без чекпоинта).

Критерий ТЗ: fingerprint химии не меняется при переключении режима integrated.
Проверяем:
  1. OrganismPhysiology создан при рождении; enable_integrated НЕ заменяет объект
     (id химии стабилен) и НЕ меняет его fingerprint на свежем организме.
  2. fp химии одинаков для integrated=False (v19) и integrated=True (v20) на свежих
     организмами (mode-invariance).
  3. Дефект (b): снимок родительского NeuroChemistry/ChemState, восстановленный
     поверх OrganismPhysiology/V20ChemState, сохраняет поля подкласса (zone_atp,
     phase, da_phasic) и переносит состояние родителя.
  4. v20 roundtrip: export -> load химии побайтно идентичен.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"C:/Users/yarma/Desktop/Ziwago18")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

import torch  # noqa: E402

torch.set_num_threads(1)

from test_v19_spiking_language import tiny_core  # noqa: E402
from test_organism_load import state_fingerprint  # noqa: E402
from ziwago.cortex.moe_cortex import export_runtime_tree, load_runtime_tree  # noqa: E402
from ziwago.organism.physiology import OrganismPhysiology, V20ChemState  # noqa: E402
from ziwago.chemistry.modulators import NeuroChemistry  # noqa: E402
from ziwago.config import ChemistryCfg  # noqa: E402


def chem_fp(core):
    return state_fingerprint(export_runtime_tree(core.chemistry))


def main():
    results = {}

    # 1. Single owner: enable_integrated does not replace the chemistry object.
    core = tiny_core()
    assert isinstance(core.chemistry, OrganismPhysiology), "birth owner must be OrganismPhysiology"
    id_before = id(core.chemistry)
    fp_before = chem_fp(core)
    core.enable_integrated_organism()
    id_after = id(core.chemistry)
    fp_after = chem_fp(core)
    results["1_single_owner_id_stable"] = (id_before == id_after)
    results["1_fp_unchanged_on_switch"] = (fp_before == fp_after)
    results["1_migration_reinitialized_false"] = (core.life.migration["chemistry_reinitialized"] is False)
    results["1_fp_before"] = fp_before[:16]
    results["1_fp_after"] = fp_after[:16]

    # 2. Mode-invariance across two fresh organisms (v19 vs v20).
    v19 = tiny_core()                       # integrated=False semantics: no life
    v20 = tiny_core()
    v20.enable_integrated_organism()        # integrated=True
    fp_v19 = chem_fp(v19)
    fp_v20 = chem_fp(v20)
    results["2_v19_owner"] = type(v19.chemistry).__name__
    results["2_v20_owner"] = type(v20.chemistry).__name__
    results["2_mode_invariant_fp"] = (fp_v19 == fp_v20)
    results["2_fp_v19"] = fp_v19[:16]
    results["2_fp_v20"] = fp_v20[:16]
    results["2_v19_has_life"] = hasattr(v19, "life")
    results["2_v20_has_life"] = hasattr(v20, "life")

    # 3. Defect (b): parent-class snapshot onto live subclass preserves v20 fields.
    parent = NeuroChemistry(ChemistryCfg())
    parent.s.da = 0.9
    parent.s.ado = 0.42
    parent.is_awake = False
    parent_snap = export_runtime_tree(parent)
    live = OrganismPhysiology(2)
    restored = load_runtime_tree(live, parent_snap)
    results["3_restored_is_same_object"] = (restored is live)
    results["3_restored_is_subclass"] = isinstance(restored, OrganismPhysiology)
    results["3_parent_state_da"] = float(restored.s.da)
    results["3_parent_state_ado"] = float(restored.s.ado)
    results["3_parent_is_awake"] = bool(restored.is_awake)
    results["3_v20_zone_atp_preserved"] = hasattr(restored, "zone_atp") and tuple(restored.zone_atp.shape) == (2,)
    results["3_v20_phase_preserved"] = getattr(restored, "phase", None) == "awake"
    results["3_v20_da_phasic_preserved"] = isinstance(restored.s, V20ChemState) and hasattr(restored.s, "da_phasic")

    # 4. v20 chemistry roundtrip byte-identical.
    body = tiny_core()
    body.enable_integrated_organism()
    body.step(torch.ones(64) * 0.2, 5)
    body.chemistry.reward(0.7)
    src = export_runtime_tree(body.chemistry)
    fp_src = state_fingerprint(src)
    fresh_owner = OrganismPhysiology(body.moe_cortex.n_columns)
    rt = load_runtime_tree(fresh_owner, src)
    fp_rt = state_fingerprint(export_runtime_tree(rt))
    results["4_v20_roundtrip_byte_identical"] = (fp_src == fp_rt)
    results["4_fp_src"] = fp_src[:16]
    results["4_fp_rt"] = fp_rt[:16]

    print("=== B.3 chemistry single-owner probe ===")
    for k, v in results.items():
        print(f"{k} = {v}")

    checks = [
        results["1_single_owner_id_stable"],
        results["1_fp_unchanged_on_switch"],
        results["1_migration_reinitialized_false"],
        results["2_mode_invariant_fp"],
        results["2_v19_owner"] == "OrganismPhysiology",
        results["2_v20_owner"] == "OrganismPhysiology",
        results["3_restored_is_same_object"],
        results["3_restored_is_subclass"],
        results["3_parent_state_da"] == 0.9,
        results["3_parent_state_ado"] == 0.42,
        results["3_parent_is_awake"] is False,
        results["3_v20_zone_atp_preserved"],
        results["3_v20_phase_preserved"],
        results["3_v20_da_phasic_preserved"],
        results["4_v20_roundtrip_byte_identical"],
    ]
    ok = all(checks)
    print(f"RESULT B.3 all_checks_pass={ok} ({sum(bool(c) for c in checks)}/{len(checks)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())