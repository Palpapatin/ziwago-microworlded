"""Диагностика загрузки чекпоинта в рантайм v20: где именно уходит память."""
import os
import sys
import time
import faulthandler
from pathlib import Path

ROOT = Path(r"C:\Users\yarma\Desktop\Ziwago18")
sys.path.insert(0, str(ROOT))

faulthandler.enable()

import torch  # noqa: E402
from ziwago_assistant import ZiwagoBiologicalSNNAssistant  # noqa: E402

CKPT = Path(r"D:\Ziwago18-storage\Produce\checkpoints\ziwago_v19_bankfixed_20261109.pt")
VERSION = int(os.environ.get("RUNTIME_VERSION", "20"))


def rss_gb() -> float:
    """Рабочая память САМОГО процесса, ГБ."""
    import ctypes
    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters),
                                                 ctypes.sizeof(counters)):
        return round(int(counters.WorkingSetSize) / 1024 / 1024, 3)
    return 0.0


def step(label: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] RSS={rss_gb():.2f} ГБ  {label}", flush=True)


def main() -> int:
    step("старт")
    t0 = time.perf_counter()
    try:
        assistant = ZiwagoBiologicalSNNAssistant(root=ROOT, checkpoint_path=CKPT, workers=1,
                                                cpu_threads=4, seed=20261109, runtime_version=VERSION)
    except Exception as exc:                                # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"ОШИБКА ЗАГРУЗКИ (v{VERSION}): {type(exc).__name__}: {exc}", flush=True)
        step("после ошибки")
        return 1
    step(f"модель загружена (v{VERSION}) за {time.perf_counter()-t0:.1f} с")
    core = assistant.organism
    print(f"  life={hasattr(core, 'life')} chemistry={type(core.chemistry).__name__} "
          f"phase={getattr(core.chemistry, 'phase', None)} vocab={len(core.dendritic_plateau_readout.col_to_tok)}",
          flush=True)
    step("готово")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
