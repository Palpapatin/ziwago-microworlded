"""Сборка markdown-карты модулей из измерений module_map_probe.

Входы:
  * module_map_fresh_*.json — результат прогона (файлы, вызовы по фазам);
  * class_instantiation_*.json — статическая карта конструирования.

Правила классификации (по измерению, не по догадке):
  SHIPPED — есть вызовы в фазе train/chat/generate;
  PHANTOM — класс создан (экземпляр в графе объектов), но ни одна его функция
            не вызвана в живом пути;
  OFF     — точка конструирования есть, но рядом флаг `*_enabled = False`
            (по умолчанию выключен);
  DEAD    — нет внешней точки конструирования вовсе.

Запуск:
    python tools/probes/build_module_map.py --map <map.json> --classes <classes.json> --out <md>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[2]

# Ручные пометки «что это за модуль» — только заголовки, без интерпретаций.
ROLE_HINTS: Dict[str, str] = {
    "ziwago/action/basal_ganglia.py": "маршрутизация действия (D1, выбора ответа)",
    "ziwago/action/contextual.py": "контекстная политика действия",
    "ziwago/app/corpora.py": "поток диалоговых корпусов",
    "ziwago/app/curriculum.py": "учебная программа",
    "ziwago/app/tokenizer.py": "BPE-токенизатор",
    "ziwago/chemistry/agents.py": "химические агенты (аденозин, лактат…)",
    "ziwago/chemistry/modulators.py": "скалярные модуляторы (DA/NE/ACh/5-HT)",
    "ziwago/chemistry/volume.py": "объёмный обмен метаболитов",
    "ziwago/cognition/dialogue.py": "состояние диалога и эпизодическая история",
    "ziwago/cognition/language/content_form.py": "композитор content/form (не подключён)",
    "ziwago/cognition/language/predictive_readout.py": "предиктивный корковый ридеут",
    "ziwago/cognition/language/readout.py": "дендритно-плато ридеут + щит метапластичности",
    "ziwago/cognition/language/syntax.py": "BA4/BA44 синтаксический мост",
    "ziwago/cognition/language/tdr.py": "TDR — выбор кандидат-индекса",
    "ziwago/cognition/memory/role_filler.py": "биндер ролей (θ-γ, подстановка заполнителя)",
    "ziwago/config.py": "конфигурации",
    "ziwago/consciousness/attention_schema.py": "схема внимания",
    "ziwago/consciousness/meta_head.py": "мета-голова (сознание)",
    "ziwago/consciousness/metrics.py": "метрики сознания (дубликат next16_core)",
    "ziwago/consciousness/workspace.py": "глобальное рабочее пространство",
    "ziwago/core/criticality.py": "активная критичность (σ≈1)",
    "ziwago/core/dendrite.py": "дендритное дерево",
    "ziwago/core/e2e_predictive.py": "предиктивное кодирование e2e",
    "ziwago/core/interoception.py": "интероцепция",
    "ziwago/core/multirate.py": "Benna–Fusi и иерархия мини-колонок",
    "ziwago/core/neuron.py": "пул спайковых нейронов (LIF)",
    "ziwago/core/sdr.py": "SDR/кодировщики",
    "ziwago/core/synapse.py": "синапс и STDP",
    "ziwago/cortex/astrocyte.py": "астроцитный микродомен",
    "ziwago/cortex/column.py": "корковая колонка",
    "ziwago/cortex/dcc.py": "DCC (готовность коры)",
    "ziwago/cortex/document_qa.py": "QA по документу (grounded)",
    "ziwago/cortex/genesis_engine.py": "движок генезиса (схемы, роли, эпос)",
    "ziwago/cortex/hierarchy.py": "корковая иерархия",
    "ziwago/cortex/lfp.py": "эфаптический LFP",
    "ziwago/cortex/moe_cortex.py": "MoE-кора, часы, матричные операции",
    "ziwago/cortex/next16_core.py": "BCM, Ларкум, щит, TDR-роутер, дашборд",
    "ziwago/cortex/son_reflection.py": "SON-рефлекторная колонка",
    "ziwago/cortex/spiking_language.py": "спайковый языковой цикл",
    "ziwago/cortex/tiled_glia.py": "тайлованная глия (ACL)",
    "ziwago/embodied/arena.py": "арена телесного поиска",
    "ziwago/emotion/limbic.py": "лимбический контур",
    "ziwago/evaluation/gates.py": "приёмочные ворота",
    "ziwago/homeostasis/controller.py": "гомеостатический контроллер",
    "ziwago/learning/ewc.py": "EWC-регуляризатор",
    "ziwago/learning/local_fisher.py": "fisher-anchor",
    "ziwago/learning/mca.py": "метапластическая чувствительность",
    "ziwago/learning/phase.py": "фазовое обучение",
    "ziwago/learning/predictive.py": "предиктивная голова",
    "ziwago/learning/surrogate.py": "суррогатные градиенты (запрещено в живом языке)",
    "ziwago/learning/sync.py": "синхронизация пол半球",
    "ziwago/memory/dual_hippocampus.py": "дуальный гиппокамп (за флагом)",
    "ziwago/memory/hippocampus.py": "спайковый гиппокамп EC→DG→CA3→CA1",
    "ziwago/memory/laplace.py": "спектр лапласовских шкал",
    "ziwago/memory/nsm.py": "нарративная самомодель",
    "ziwago/memory/phase_vsa.py": "θ-γ биндер историй",
    "ziwago/memory/tcc.py": "тораидальный аттрактор отношений",
    "ziwago/memory/working.py": "рабочая память",
    "ziwago/motor/pool.py": "моторный пул",
    "ziwago/motor/syllabary_decoder.py": "декодер слогов",
    "ziwago/organism/admc.py": "аллостатический контроллер любопытства",
    "ziwago/organism/flat.py": "плоское ядро организма (монолит)",
    "ziwago/organism/integrated.py": "интегрированный контур v20",
    "ziwago/organism/physiology.py": "физиология (единственный владелец химии)",
    "ziwago/organism/sleep.py": "сон-обвязка",
    "ziwago/organism/verified.py": "проверяемый контур (evidence/authority)",
    "ziwago/organism_core.py": "обёртка ZiwagoOrganism (когнитивный цикл)",
    "ziwago/thalamus/relay.py": "таламус-реле",
    "ziwago/thalamus/trn.py": "таламо-корковой TRN",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.map).read_text(encoding="utf-8"))
    classes = json.loads(Path(args.classes).read_text(encoding="utf-8"))

    # классы → файлы
    class_to_files: Dict[str, set] = {}
    for name, info in classes["classes"].items():
        for path in info["defined_in"]:
            class_to_files.setdefault(path, set()).add(name)

    # файлы с флагом False рядом с конструированием
    flagged: Dict[str, list] = {}
    for name, info in classes["classes"].items():
        for site in info["external_instantiations"]:
            for flag in site["flags_nearby"]:
                if flag.endswith("=False"):
                    flagged.setdefault(site["at"].split(":")[0], []).append(f"{name}:{flag}")

    rows = []
    for rel, info in sorted(data["files"].items()):
        calls = info["functions_called"]
        phases = info.get("calls_by_phase", {})
        instances = list(info["instances"].keys())
        if calls > 0:
            status = "SHIPPED"
        elif instances:
            status = "PHANTOM"
        elif rel in flagged:
            status = "OFF"
        else:
            status = "DEAD"
        rows.append({
            "file": rel,
            "lines": info["lines"],
            "functions": info["functions"],
            "status": status,
            "calls": calls,
            "train": phases.get("train", 0),
            "chat": phases.get("chat", 0),
            "generate": phases.get("generate", 0),
            "instances": instances,
            "role": ROLE_HINTS.get(rel, ""),
            "flags": flagged.get(rel, []),
        })

    summary: Dict[str, Dict[str, int]] = {}
    for row in rows:
        bucket = summary.setdefault(row["status"], {"files": 0, "lines": 0})
        bucket["files"] += 1
        bucket["lines"] += row["lines"]

    # ── второй уровень: по МЕХАНИЗМАМ (классы) ─────────────────────────────
    mech_rows = []
    for name, info in classes["classes"].items():
        path = info["defined_in"][0]
        file_info = data["files"].get(path)
        if file_info is None:
            continue
        # пропускаем чистые dataclass-ы/конфиги/результаты
        if name.endswith(("Cfg", "Config", "Out", "State", "Event", "Proposal", "Edge",
                          "Item", "Group", "Decision", "Level", "Result", "Record",
                          "Query", "Plan", "Spec", "View", "Options", "Params")):
            continue
        if file_info["functions"] == 0:
            continue
        instances = [cls for cls in file_info["instances"] if cls.endswith("." + name)]
        guards = [g for site in info["external_instantiations"] for g in site.get("guards", [])]
        off_guard = [g for g in guards if "enabled" in g.lower()]
        if file_info["functions_called"] > 0 and instances:
            status = "SHIPPED"
        elif off_guard:
            status = "OFF"
        elif instances:
            status = "PHANTOM"
        elif guards:
            status = "OFF"
        else:
            status = "DEAD"
        mech_rows.append({
            "name": name, "file": path, "status": status,
            "instances": len(instances),
            "calls": file_info["functions_called"],
            "guard": "; ".join(off_guard[:2]) or ("; ".join(guards[:1]) if guards else ""),
        })
    mech_rows.sort(key=lambda r: (r["status"], r["file"], r["name"]))

    out = ["# Карта модулей Ziwago18 — «заявлено vs работает»", ""]
    out.append("Сгенерировано из измерений: `tools/probes/module_map_probe.py` → "
               "`tools/probes/build_module_map.py`. Классификация НЕ выдумана: "
               "статус выведен из числа реальных вызовов функций в фазах "
               "обучение / чат / прямая генерация.")
    out.append("")
    out.append("## Уровень 1 — файлы")
    out.append("")
    out.append("| Статус | Файлов | Строк |")
    out.append("|---|---:|---:|")
    for status in ("SHIPPED", "PHANTOM", "OFF", "DEAD"):
        bucket = summary.get(status, {"files": 0, "lines": 0})
        out.append(f"| {status} | {bucket['files']} | {bucket['lines']} |")
    total_files = sum(b["files"] for b in summary.values())
    total_lines = sum(b["lines"] for b in summary.values())
    out.append(f"| **Итого** | **{total_files}** | **{total_lines}** |")
    out.append("")
    out.append("| Файл | Строк | Статус | Вызовы (train/chat/gen) | Экземпляров | Роль |")
    out.append("|---|---:|---|---:|---:|---|")
    for row in rows:
        inst = ", ".join(cls.split(".")[-1] for cls in row["instances"][:3]) or "—"
        role = row["role"] or "—"
        out.append(f"| `{row['file']}` | {row['lines']} | **{row['status']}** | "
                   f"{row['train']}/{row['chat']}/{row['generate']} | {inst} | {role} |")
    out.append("")

    out.append("## Уровень 2 — механизмы (классы)")
    out.append("")
    out.append("| Механизм | Файл | Статус | Экземпляров | Guard (условие включения) |")
    out.append("|---|---|---|---:|---|")
    for row in mech_rows:
        out.append(f"| `{row['name']}` | `{row['file']}` | **{row['status']}** | "
                   f"{row['instances']} | {row['guard'] or '—'} |")
    out.append("")
    mech_summary: Dict[str, int] = {}
    for row in mech_rows:
        mech_summary[row["status"]] = mech_summary.get(row["status"], 0) + 1
    out.append("Итого по механизмам: " + ", ".join(
        f"{k} — {v}" for k, v in sorted(mech_summary.items())))
    out.append("")

    Path(args.out).write_text("\n".join(out), encoding="utf-8")
    print(f"записано: {args.out}")
    for status in ("SHIPPED", "PHANTOM", "OFF", "DEAD"):
        bucket = summary.get(status, {"files": 0, "lines": 0})
        print(f"  {status}: {bucket['files']} файлов, {bucket['lines']} строк")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
