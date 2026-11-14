"""Кто РЕАЛЬНО читает файлы в рантайме (open/read_text/load), а не просто упоминает."""
import glob
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent

READERS = re.compile(
    r"(?:open|read_text|load|Path\()\s*\(?\s*[^)\n]{0,80}?([A-Za-z0-9_./\\-]+\.(?:txt|md|html|docx))",
    re.I)

code = [f for f in glob.glob("*.py") + glob.glob("tools/*.py") + glob.glob("tests/*.py")
        if "pycache" not in f]

hits = {}
for f in code:
    s = Path(f).read_text(encoding="utf-8", errors="replace")
    for line in s.splitlines():
        for m in READERS.finditer(line):
            name = m.group(1)
            if name.startswith(("data/", "Produce/", "tests/", "tools/")):
                continue
            hits.setdefault(name, []).append((f, line.strip()[:100]))

print("--- ФАЙЛЫ, ЧИТАЕМЫЕ В РАНТАЙМЕ ---")
for name, where in sorted(hits.items()):
    print("   {:<34} <- {}".format(name, where[0][0]))
    for f, ln in where[:2]:
        print("        {}: {}".format(f[:44], ln))

print()
print("--- УПОМИНАНИЯ БЕЗ ЧТЕНИЯ (можно удалять) ---")
MENT = re.compile(r"[\"']([A-Za-z0-9_./-]+\.(?:txt|md|html|docx))[\"']")
for f in code:
    s = Path(f).read_text(encoding="utf-8", errors="replace")
    for m in MENT.finditer(s):
        n = m.group(1)
        if n in hits:
            continue
        if n.endswith("merges.txt") or n.endswith("vocab.json"):
            continue
        print("   {:<32} <- {}".format(n, f[:52]))
