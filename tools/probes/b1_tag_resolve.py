"""B.1 diag v9: проверить, что каждый neuroweave10-тег резолвится в ziwago.

diag8: 31 disallowed тег (296 вхождений), все neuroweave10.*. neuroweave10/
удалён — перенесён в ziwago/ (структура подмодулей совпадает). Гипотеза:
замена префикса neuroweave10. -> ziwago. резолвит все классы.

Прибор (READONLY load + importlib, БЕЗ конструкции): для каждого bad-тега
import_module(ziwago-модуль) + getattr(name). Печатает RESOLVED/MISSING.
Это доказательство карты миграции перед правкой load_runtime_tree.
"""
from __future__ import annotations
import faulthandler
import importlib
import mmap
import sys
import time
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
    print(f"[diag9 +{time.perf_counter()-_t0:6.1f}s] {m}", flush=True)


def main():
    payload = torch.load(str(CKPT), map_location="cpu", weights_only=True, mmap=True)
    log("payload loaded READONLY")
    bad = set()

    def walk(o):
        if isinstance(o, dict):
            if "__object__" in o:
                module, name = o["__object__"].split(":", 1)
                if not allowed(module, name):
                    bad.add((module, name))
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    for top in ("core_runtime_state", "v19_state"):
        if top in payload:
            walk(payload[top])
    log(f"distinct disallowed tags = {len(bad)}")

    resolved = 0
    missing = []
    for module, name in sorted(bad):
        if not module.startswith("neuroweave10."):
            missing.append((module, name, "not neuroweave10 prefix"))
            continue
        newmod = "ziwago." + module[len("neuroweave10."):]
        try:
            m = importlib.import_module(newmod)
            cls = getattr(m, name)
            assert isinstance(cls, type), f"{name} is not a class"
            log(f"  RESOLVED {module}:{name} -> {newmod}:{name}")
            resolved += 1
        except Exception as exc:  # noqa: BLE001
            missing.append((module, name, f"{type(exc).__name__}: {exc}"))
            log(f"  MISSING  {module}:{name} -> {newmod}: {type(exc).__name__}: {exc}")
    log(f"RESOLVED={resolved}/{len(bad)} MISSING={len(missing)}")
    for module, name, why in missing:
        log(f"  UNRESOLVED {module}:{name} :: {why}")
    log("DIAG9 DONE")
    return 0 if not missing else 2


if __name__ == "__main__":
    try:
        code = main()
    except BaseException as exc:  # noqa: BLE001
        import traceback
        log(f"FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        code = 1
    raise SystemExit(code)