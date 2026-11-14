"""C1: разбор причин, по которым строки куррикулы не допущены к обучению."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

for tag, path in (("scaled3 (моя смесь v4, 300 строк)",
                   r"D:\Ziwago18-storage\Produce\v19_scaled3_20261109.json"),
                  ("bankfixed (v3, 150 строк)",
                   r"D:\Ziwago18-storage\Produce\v19_bankfixed_20261109.json")):
    f = Path(path)
    if not f.exists():
        print(f"{tag}: отчёта нет")
        continue
    d = json.loads(f.read_text(encoding="utf-8"))
    pilot = d.get("pilot") or {}
    filtering = pilot.get("filtering") or []
    reasons = Counter(item.get("reason") for item in filtering)
    print("=" * 92)
    print(tag)
    print(f"  отобрано: {pilot.get('selected_records')} | отфильтровано: "
          f"{pilot.get('filtered_records')} | обучено: {pilot.get('trained_records')} "
          f"| фактических волн: {pilot.get('actual_waves')}")
    print("  причины отказа:")
    for reason, count in reasons.most_common():
        print(f"    {str(reason):<28} {count}")
    # какие источники отсеиваются сильнее
    per_source = Counter()
    lines = {item.get("source_line") for item in filtering}
    ledger = d.get("exposure_ledger") or []
    for row in ledger:
        if row.get("source_line") in lines:
            per_source[str(row.get("source") or row.get("family_id", "?"))[:38]] += 1
    print("  отсеяно по источникам:")
    for source, count in per_source.most_common():
        print(f"    {source:<40} {count}")
    # банк и качество обучения
    counts = pilot.get("learning_counts") or {}
    print(f"  банк: обновлений {counts.get('bank_token_updates')}, "
          f"отказов по бюджету {counts.get('bank_memory_budget_rejections')}, "
          f"всего отказов {counts.get('bank_token_rejections')}, "
          f"pruned {counts.get('bank_pruned_entries')}")
    print(f"  шортлист PC: recall {counts.get('pc_shortlist_recall')}, "
          f"расширений {counts.get('pc_target_extensions')}, "
          f"шаг срезан у {counts.get('readout_step_limited')} обновлений")
