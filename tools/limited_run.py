#!/usr/bin/env python3
"""Запуск тяжёлого процесса с ЖЁСТКИМИ лимитами памяти и CPU.

Зачем: на этой машине (15.9 ГБ RAM, GPU делится с браузером) тяжёлый прогон,
запущенный без ограничения, съедает всю память и приводит к OOM у всей
системы — включая чужую работу. Windows даёт нужный механизм: Job Object с
лимитом памяти процесса. При его срабатывании процесс получает MemoryError
внутри себя и умирает, а система остаётся живой.

Использование:
    python tools/limited_run.py --ram-pct 60 --cpu-pct 60 --log out.log -- python tools/check_v20_generation.py

Лимиты по умолчанию берутся из переменных окружения ZIWAGO_RAM_PCT / ZIWAGO_CPU_PCT.
Пороги (85 %) — из правил проекта: VRAM ≤ 85 %, RAM ≤ 85 %, CPU ≤ 85 % ядер;
здесь по умолчанию 60 %, чтобы оставить запас пользователю.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import time
from ctypes import wintypes

# --- константы Windows Job Object ---
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_CPU_RATE_CONTROL_INFORMATION = 15
CPU_RATE_CONTROL_ENABLE_HARD_CAP = 0x00000001
CREATE_SUSPENDED = 0x00000004

class IO_COUNTERS(ctypes.Structure):
    _fields_ = [("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong)]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


class JOBOBJECT_CPU_RATE_CONTROL_INFORMATION(ctypes.Structure):
    class RATE_CONTROL(ctypes.Union):
        _fields_ = [("CpuRate", wintypes.DWORD), ("Weight", wintypes.DWORD)]

    _fields_ = [("ControlFlags", wintypes.DWORD), ("RateControl", RATE_CONTROL)]


def total_physical_memory() -> tuple[int, int]:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    kernel32 = ctypes.windll.kernel32
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
    kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError(kernel32.GetLastError(), "GlobalMemoryStatusEx failed")
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def cpu_count() -> int:
    return os.cpu_count() or 4


def make_job(ram_bytes: int, cpu_pct: int):
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, wintypes.INT, wintypes.LPVOID, wintypes.DWORD]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.GetLastError.restype = wintypes.DWORD
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(kernel32.GetLastError(), "CreateJobObjectW failed")
    limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    limits.BasicLimitInformation.LimitFlags = (
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_JOB_MEMORY
        | JOB_OBJECT_LIMIT_PROCESS_MEMORY)
    limits.JobMemoryLimit = ram_bytes
    limits.ProcessMemoryLimit = ram_bytes
    try:
        if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(limits),
                                                ctypes.sizeof(limits)):
            raise OSError(kernel32.GetLastError(), "SetInformationJobObject(JobMemoryLimit) failed")
        if cpu_pct < 100:
            cpu = JOBOBJECT_CPU_RATE_CONTROL_INFORMATION()
            cpu.ControlFlags = CPU_RATE_CONTROL_ENABLE_HARD_CAP
            cpu.RateControl.CpuRate = max(1, int(cpu_pct * 100))
            if not kernel32.SetInformationJobObject(
                    job, JOB_OBJECT_CPU_RATE_CONTROL_INFORMATION,
                    ctypes.byref(cpu), ctypes.sizeof(cpu)):
                raise OSError(kernel32.GetLastError(), "SetInformationJobObject(CPU hard cap) failed")
    except OSError:
        kernel32.CloseHandle(job)
        raise
    return job


def assign(job, pid: int) -> tuple[bool, int]:
    """Назначает процесс в job object. Процесс должен быть создан приостановленным."""
    kernel32 = ctypes.windll.kernel32
    PROCESS_SET_QUOTA = 0x0100
    PROCESS_TERMINATE = 0x0001
    PROCESS_QUERY_INFORMATION = 0x0400
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.GetLastError.restype = wintypes.DWORD
    handle = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE | PROCESS_QUERY_INFORMATION,
                                   False, pid)
    if not handle:
        return False, kernel32.GetLastError()
    ok = bool(kernel32.AssignProcessToJobObject(job, handle))
    error = kernel32.GetLastError() if not ok else 0
    kernel32.CloseHandle(handle)
    return ok, error


def start_in_job(command: list[str], job, *, stdout, stderr):
    process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                               creationflags=CREATE_SUSPENDED)
    try:
        ok, error = assign(job, process.pid)
    except BaseException:
        process.kill()
        process.wait()
        raise
    if not ok:
        process.kill()
        process.wait()
        raise RuntimeError(
            f"AssignProcessToJobObject failed with Win32 {error}; suspended workload was terminated"
        )
    ntdll = ctypes.windll.ntdll
    ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
    ntdll.NtResumeProcess.restype = ctypes.c_long
    try:
        status = ntdll.NtResumeProcess(process._handle)
    except BaseException:
        process.kill()
        process.wait()
        raise
    if status != 0:
        process.kill()
        process.wait()
        raise RuntimeError(
            f"NtResumeProcess returned {status}; suspended workload was terminated"
        )
    return process


def job_peak(job) -> tuple[int, int]:
    """Пик памяти процесса в job и жёсткий лимит (байты)."""
    kernel32 = ctypes.windll.kernel32
    kernel32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE, wintypes.INT, wintypes.LPVOID, wintypes.DWORD, wintypes.LPVOID]
    kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    kernel32.GetLastError.restype = wintypes.DWORD
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    ok = kernel32.QueryInformationJobObject(job, 9, ctypes.byref(info),
                                            ctypes.sizeof(info), None)
    if not ok:
        raise OSError(kernel32.GetLastError(), "QueryInformationJobObject failed")
    return int(info.PeakJobMemoryUsed), int(info.JobMemoryLimit)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ram-pct", type=int,
                        default=int(os.environ.get("ZIWAGO_RAM_PCT", "60")))
    parser.add_argument("--ram-free-pct", type=int,
                        default=int(os.environ.get("ZIWAGO_RAM_FREE_PCT", "95")),
                        help="Потолок как доля СВОБОДНОЙ памяти в момент запуска. "
                             "Это и есть настоящая защита машины: лимит от общего "
                             "объёма не защищает, если память уже занята чужими "
                             "приложениями (у пользователя занято ~7 ГБ).")
    parser.add_argument("--cpu-pct", type=int,
                        default=int(os.environ.get("ZIWAGO_CPU_PCT", "60")))
    parser.add_argument("--log", type=str, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = [part for part in args.command if part != "--"]
    if not command:
        parser.error("no command given")
    if os.name != "nt":
        parser.error("limited_run requires Windows Job Objects and is not supported on this platform")
    if not 5 <= args.ram_pct <= 95:
        parser.error("--ram-pct must be between 5 and 95")
    if not 5 <= args.ram_free_pct <= 100:
        parser.error("--ram-free-pct must be between 5 and 100")
    if not 1 <= args.cpu_pct <= 100:
        parser.error("--cpu-pct must be between 1 and 100")
    total, avail = total_physical_memory()
    cap_total = int(total * args.ram_pct / 100)
    cap_free = int(avail * args.ram_free_pct / 100)
    cap = min(cap_total, cap_free)
    out = open(args.log, "ab", buffering=0) if args.log else None
    started = time.perf_counter()

    def report(line: str) -> None:
        # Консоль Windows по умолчанию cp1251 — символ «←» роняет печать и
        # ТЕРЯЕТ весь отчёт о прогоне. Печатаем безопасно, в лог — в UTF-8.
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"), flush=True)
        if out is not None:
            out.write((line + "\n").encode("utf-8"))

    try:
        job = make_job(cap, args.cpu_pct)
    except OSError as exc:
        report(f"[limited_run] failed to configure Job Object: {exc}")
        if out is not None:
            out.close()
        return 3
    report(f"[limited_run] RAM cap {cap/1e9:.2f} ГБ "
           f"(min({args.ram_pct}% от {total/1e9:.1f} ГБ, {args.ram_free_pct}% от свободных "
           f"{avail/1e9:.1f} ГБ)), CPU cap {args.cpu_pct}% "
           f"({max(1, args.cpu_pct*cpu_count()//100)} из {cpu_count()} ядер)")
    try:
        # The child remains suspended until the Job Object limits are active.
        process = start_in_job(
            command, job, stdout=out or subprocess.DEVNULL,
            stderr=subprocess.STDOUT if out else subprocess.DEVNULL)
    except OSError as exc:
        report(f"[limited_run] failed to start: {exc}")
        ctypes.windll.kernel32.CloseHandle(job)
        if out is not None:
            out.close()
        return 2
    except RuntimeError as exc:
        report(f"[limited_run] {exc}")
        ctypes.windll.kernel32.CloseHandle(job)
        if out is not None:
            out.close()
        return 3

    try:
        code = process.wait()
        elapsed = time.perf_counter() - started
        peak, limit = job_peak(job)
        report(f"[limited_run] завершился с кодом {code} за {elapsed/60:.1f} мин; "
               f"пик памяти {peak/1e9:.2f} ГБ при лимите {limit/1e9:.2f} ГБ"
               + ("  ← УБИТ ЛИМИТОМ ПАМЯТИ" if code != 0 and peak >= limit * 0.95 else ""))
        return code
    finally:
        ctypes.windll.kernel32.CloseHandle(job)
        if out is not None:
            out.close()


if __name__ == "__main__":
    raise SystemExit(main())
