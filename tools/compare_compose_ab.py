import json
import sys
from pathlib import Path

PRODUCE = Path(r"D:\Ziwago18-storage\Produce")
ARMS = [
    ("A  bankfixed, apical OFF (контроль)", "v19_compose_A_bankfixed_apical_off_20261109.json"),
    ("B  apical2,  apical ON  (лечение)", "v19_compose_B_apical_on_20261109.json"),
    ("C  apical2,  apical OFF (изоляция весов)", "v19_compose_C_apical_off_20261109.json"),
]
CONDS = ("replay_literal_prefix", "compose_prefix", "compose_question", "compose_question_prefix")


def load(name):
    path = PRODUCE / name
    if not path.is_file():
        return None
    return json.load(open(path, encoding="utf-8"))


def main():
    print(f"{'условие':26s} " + " ".join(f"{tag:>10s}" for tag, _ in ARMS))
    data = {}
    for tag, name in ARMS:
        report = load(name)
        data[tag] = report
    for cond in CONDS:
        cells = []
        for tag, _ in ARMS:
            report = data[tag]
            if report is None:
                cells.append("—")
                continue
            m = report["metrics"][cond]
            cells.append(f"{m['passes']}/{m['cases']}")
        print(f"{cond:26s} " + " ".join(f"{c:>10s}" for c in cells))
    print()
    for cond in CONDS:
        print(f"--- {cond} ---")
        for tag, _ in ARMS:
            report = data[tag]
            if report is None:
                print(f"  {tag}: нет отчёта")
                continue
            m = report["metrics"][cond]
            print(f"  {tag}: passes={m['passes']}/{m['cases']} degenerate={m['degenerate']} "
                  f"eos={m['learned_eos']} conflict_mean={m['conflict_mean']}")
        print()
    print("--- образцы продолжений (compose_prefix) ---")
    for tag, _ in ARMS:
        report = data[tag]
        if report is None:
            continue
        print(f"  {tag}")
        for row in report["rows"]:
            c = row["conditions"]["compose_prefix"]
            ec = row["conditions"].get("episode_conflict_mean")
            print(f"    finish={c['finish_reason']:16s} key_phrase={c['coherence']['key_phrase_present']} "
                  f"ep_conflict={ec} :: {c['continuation'][:110]!r}")
    print()
    for tag, _ in ARMS:
        report = data[tag]
        if report is None:
            continue
        print(f"{tag}: state_fingerprints_match={report.get('state_fingerprints_match')} "
              f"core_restored={report.get('core_state_restored')} v19_restored={report.get('v19_state_restored')} "
              f"checkpoint_unchanged={report.get('checkpoint_unchanged')} training={report.get('training_performed')} "
              f"promoted={report.get('promoted')} elapsed={report.get('elapsed_seconds')}")
    for tag, _ in ARMS:
        report = data[tag]
        if report is None:
            continue
        ast = report.get("apical_state")
        if ast:
            print(f"{tag}: apical_state={json.dumps(ast, ensure_ascii=False)}")


if __name__ == "__main__":
    sys.exit(main())
