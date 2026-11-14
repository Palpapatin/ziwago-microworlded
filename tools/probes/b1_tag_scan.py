"""B.1 diag v8: найти __object__-тег, который load_runtime_tree отвергает.

Ошибка 'Unsupported runtime class tag' (moe_cortex.py:133). Allowlist модулей:
ziwago. / train_ziwago_ / torch.nn. / types / __main__+{4 имени}. Чекпоинт
2026-11-09 мог сохранить класс из модуля, переехавшего при рефакторинге v21.

Сканер (БЕЗ конструкции, READONLY mmap, дёшево): обходит core_runtime_state и
v19_state, собирает ВСЕ __object__-теги, считает по префиксам модуля, помечает
те, что НЕ проходят allowlist. Печатает точные module:ClassName нарушителей.
"""
from __future__ import annotations
import faulthandler
import mmap
import sys
import time
from collections import Counter
from pathlib import Path

faulthandler.enable(all_threads=True)
ROOT = Path(r"C:/Users/yarma/Desktop/Ziwago18")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
torch.set_num_threads(4)

CKPT = Path(r"D:/Ziwago18-storage/Produce/checkpoints/ziwago_v19_apical2_20261109.pt")
_t0 = time.perf_counter()
_KEEPALIVE = []
_MAIN_OK = {"ThetaGammaRoleFillerBinder", "ConversationalDialogueState",
            "DendriticPlateauMatrixReadout", "BA44AttractorSyntaxBridge"}


def _readonly_from_file(filename, shared, size):
    f = open(filename, "rb")
    mm = mmap.mmap(f.fileno(), size, access=mmap.ACCESS_READ)
    _KEEPALIVE.append((f, mm))
    return torch.frombuffer(mm, dtype=torch.uint8).untyped_storage()


torch.UntypedStorage.from_file = _readonly_from_file


def allowed(module, name):
    return (module.startswith("ziwago.") or module.startswith("train_ziwago_")
            or module.startswith("torch.nn.") or module == "types"
            or (module == "__main__" and name in _MAIN_OK))


def log(m):
    print(f"[diag8 +{time.perf_counter()-_t0:6.1f}s] {m}", flush=True)


def main():
    payload = torch.load(str(CKPT), map_location="cpu", weights_only=True, mmap=True)
    log("payload loaded READONLY")
    tags = Counter()
    bad = {}

    def walk(o, path):
        if isinstance(o, dict):
            if "__object__" in o:
                tag = o["__object__"]
                tags[tag] += 1
                module, name = tag.split(":", 1)
                if not allowed(module, name):
                    bad.setdefault(tag, path)
            for k, v in o.items():
                walk(v, f"{path}.{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")

    for top in ("core_runtime_state", "v19_state"):
        if top in payload:
            walk(payload[top], top)
    log(f"total distinct __object__ tags = {len(tags)}; total occurrences = {sum(tags.values())}")
    prefixes = Counter(t.split(":", 1)[0].split(".")[0] for t in tags)
    log(f"module top-level prefixes: {dict(prefixes)}")
    log(f"DISALLOWED tags = {len(bad)}")
    for tag, path in sorted(bad.items()):
        log(f"  BAD {tag}  count={tags[tag]}  first_path={path}")
    if not bad:
        log("  (none — all tags pass allowlist)")
    log("DIAG8 DONE")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except BaseException as exc:  # noqa: BLE001
        import traceback
        log(f"FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        code = 1
    raise SystemExit(code)