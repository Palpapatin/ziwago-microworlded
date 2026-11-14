import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
for tag, p in (("v9 baseline (0.217 эфф.)",
                r"D:\Ziwago18-storage\Produce\v19_nm_low3_20261109.json"),
               ("bankfixed (150 строк, банк 2 ГиБ)",
                r"D:\Ziwago18-storage\Produce\v19_compose_bankfixed2_20261109.json")):
    f = Path(p)
    if not f.exists():
        print(tag, "— НЕТ ФАЙЛА")
        continue
    d = json.loads(f.read_text(encoding="utf-8"))
    print(tag)
    for k, v in d["metrics"].items():
        print("   %-26s %d/%d  деген=%d EOS=%d  conflict=%s" % (
            k, v["passes"], v["cases"], v["degenerate"], v["learned_eos"], v.get("conflict_mean")))
    print("   state_restored", d.get("evaluation_state_and_rng_restored"),
          "| ckpt", d.get("checkpoint_unchanged"), "| master", d.get("master_unchanged"))
    a = d.get("arbitration") or {}
    if a.get("ach") is not None:
        print("   ach в кандидате:", a.get("ach"))
